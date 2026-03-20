import torch
import torch.nn as nn
import torchvision.ops as ops


import torch
import torch.nn as nn
import torchvision.ops as ops


# ──────────────────────────────────────────────
# 1. DINOv2 Backbone
# ──────────────────────────────────────────────
class DINOv2Backbone(nn.Module):
    """
    model_name:
        "dinov2_vits14"  → embed_dim=384  (~22M param)
        "dinov2_vitb14"  → embed_dim=768  (~86M param)

    unfreeze_last_n:
        0  → hammasi frozen
        2  → oxirgi 2 blok trainable
        12 → hammasi trainable (full fine-tune, vits14 = 12 blok)
    """
    def __init__(self, model_name: str = "dinov2_vits14",
                 unfreeze_last_n: int = 12):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.embed_dim = self.model.embed_dim

        # Avval hammani freeze
        for param in self.model.parameters():
            param.requires_grad = False

        # Oxirgi N blokni unfreeze
        if unfreeze_last_n > 0:
            blocks = list(self.model.blocks)
            for block in blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True
            # norm layer ham unfreeze
            for param in self.model.norm.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input : [B, 3, 224, 224]
        Output: [B, embed_dim, 16, 16]
        """
        out = self.model.forward_features(x)
        patch_tokens = out["x_norm_patchtokens"]       # [B, 256, C]
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)                          # 16
        feat = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return feat

    def get_backbone_params(self):
        """Differential LR uchun backbone parametrlari."""
        return [p for p in self.model.parameters() if p.requires_grad]


# ──────────────────────────────────────────────
# 2. Detection Head  (kuchaytirilgan)
# ──────────────────────────────────────────────
class DetectionHead(nn.Module):
    """
    3 qatlamli neck + cls / box / centerness head.
    Centerness → kichik va nozik defektlarni yaxshiroq topadi.
    """
    def __init__(self, in_channels: int, num_classes: int, hidden: int = 256):
        super().__init__()

        self.neck = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.cls_head        = nn.Conv2d(hidden, num_classes + 1, 1)
        self.box_head        = nn.Conv2d(hidden, 4, 1)
        self.centerness_head = nn.Conv2d(hidden, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.cls_head, self.box_head, self.centerness_head]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, feat: torch.Tensor):
        """
        Output:
            cls_logits : [B, H*W, num_classes+1]
            box_preds  : [B, H*W, 4]   sigmoid → [0,1]
            centerness : [B, H*W, 1]   sigmoid → [0,1]
        """
        x   = self.neck(feat)
        cls = self.cls_head(x)
        box = torch.sigmoid(self.box_head(x))
        ctr =  self.centerness_head(x)

        B, _, H, W = cls.shape
        cls = cls.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        box = box.permute(0, 2, 3, 1).reshape(B, H * W, 4)
        ctr = ctr.permute(0, 2, 3, 1).reshape(B, H * W, 1)
        return cls, box, ctr

    def get_head_params(self):
        return list(self.parameters())


# ──────────────────────────────────────────────
# 3. To'liq model
# ──────────────────────────────────────────────
class DINOv2Detector(nn.Module):
    def __init__(self,
                 num_classes    : int = 6,
                 model_name     : str = "dinov2_vits14",
                 unfreeze_last_n: int = 12):
        super().__init__()
        self.backbone = DINOv2Backbone(model_name, unfreeze_last_n)
        self.head     = DetectionHead(
            in_channels=self.backbone.embed_dim,
            num_classes=num_classes,
        )

    def forward(self, images):
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)
        feat = self.backbone(images)
        return self.head(feat)          # cls, box, centerness

    def get_optimizer_groups(self, backbone_lr=1e-5, head_lr=1e-3, weight_decay=1e-4):
        """
        Differential LR:
            backbone → kichik LR  (nozik fine-tune)
            head     → katta LR   (yangi o'rganish)
        """
        return [
            {"params": self.backbone.get_backbone_params(),
             "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": self.head.get_head_params(),
             "lr": head_lr, "weight_decay": weight_decay},
        ]

    def count_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Jami parametr     : {total:,}")
        print(f"Trainable parametr: {trainable:,}  ({100 * trainable / total:.1f}%)")


# ──────────────────────────────────────────────
# 4. Loss  (FocalLoss + Centerness)
# ──────────────────────────────────────────────
class DetectionLoss(nn.Module):
    """
    Classification : FocalLoss  (class imbalance uchun)
    Localization   : SmoothL1
    Centerness     : BCELoss
    """
    def __init__(self, lambda_box=5.0, lambda_ctr=1.0,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.box_loss   = nn.SmoothL1Loss()
        self.ctr_loss   = nn.BCEWithLogitsLoss()
        self.lambda_box = lambda_box
        self.lambda_ctr = lambda_ctr
        self.alpha      = focal_alpha
        self.gamma      = focal_gamma

    def focal_loss(self, logits, targets):
        ce   = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt   = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()

    def compute_centerness_target(self, gt_box, cell_cx, cell_cy, img_size):
        xmin, ymin, xmax, ymax = gt_box / img_size
        l = (cell_cx - xmin).clamp(min=1e-6)
        r = (xmax - cell_cx).clamp(min=1e-6)
        t = (cell_cy - ymin).clamp(min=1e-6)
        b = (ymax - cell_cy).clamp(min=1e-6)
        return torch.sqrt(
            (torch.min(l, r) / torch.max(l, r)) *
            (torch.min(t, b) / torch.max(t, b))
        ).clamp(0, 1)

    def forward(self, cls_logits, box_preds, centerness, targets, img_size=224):
        B, N, _ = cls_logits.shape
        H = W = int(N ** 0.5)

        total_cls = torch.tensor(0.0, device=cls_logits.device)
        total_box = torch.tensor(0.0, device=cls_logits.device)
        total_ctr = torch.tensor(0.0, device=cls_logits.device)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=cls_logits.device),
            torch.arange(W, device=cls_logits.device),
            indexing="ij"
        )
        cx = (grid_x.flatten() + 0.5) / W   # [N]
        cy = (grid_y.flatten() + 0.5) / H   # [N]

        for b in range(B):
            gt_boxes  = targets[b]["boxes"].to(cls_logits.device).float()
            gt_labels = targets[b]["labels"].to(cls_logits.device).long()

            if len(gt_boxes) == 0:
                cls_target = torch.zeros(N, dtype=torch.long, device=cls_logits.device)
                total_cls += self.focal_loss(cls_logits[b], cls_target)
                continue

            gt_cx = ((gt_boxes[:, 0] + gt_boxes[:, 2]) / 2) / img_size
            gt_cy = ((gt_boxes[:, 1] + gt_boxes[:, 3]) / 2) / img_size
            gt_w  =  (gt_boxes[:, 2] - gt_boxes[:, 0]) / img_size
            gt_h  =  (gt_boxes[:, 3] - gt_boxes[:, 1]) / img_size
            gt_cxcywh = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=1)

            cell_centers = torch.stack([cx, cy], dim=1)
            gt_centers   = torch.stack([gt_cx, gt_cy], dim=1)
            dist         = torch.cdist(cell_centers, gt_centers)
            best_gt_idx  = dist.argmin(dim=1)
            min_dist     = dist.min(dim=1).values
            threshold    = 1.5 / H                  # kengroq threshold
            positive_mask = min_dist < threshold

            cls_target = torch.zeros(N, dtype=torch.long, device=cls_logits.device)
            cls_target[positive_mask] = gt_labels[best_gt_idx[positive_mask]]
            total_cls += self.focal_loss(cls_logits[b], cls_target)

            if positive_mask.sum() > 0:
                pred_pos = box_preds[b][positive_mask]
                gt_pos   = gt_cxcywh[best_gt_idx[positive_mask]]
                total_box += self.box_loss(pred_pos, gt_pos)

                ctr_pred   = centerness[b][positive_mask].squeeze(-1)
                ctr_target = torch.stack([
                    self.compute_centerness_target(
                        gt_boxes[best_gt_idx[positive_mask][i]],
                        cx[positive_mask][i],
                        cy[positive_mask][i],
                        img_size
                    )
                    for i in range(positive_mask.sum())
                ])
                total_ctr += self.ctr_loss(ctr_pred, ctr_target)

        loss = (total_cls / B
                + self.lambda_box * total_box / B
                + self.lambda_ctr * total_ctr / B)

        return loss, {
            "cls_loss": total_cls.item() / B,
            "box_loss": total_box.item() / B,
            "ctr_loss": total_ctr.item() / B,
        }




# # ──────────────────────────────────────────────
# # Sinash
# # ──────────────────────────────────────────────
# if __name__ == "__main__":
#     import torch.optim as optim

#     model = DINOv2Detector(num_classes=6,
#                            model_name="dinov2_vits14",
#                            unfreeze_last_n=12)
#     model.count_params()

#     # Differential LR optimizer
#     optimizer = optim.AdamW(
#         model.get_optimizer_groups(backbone_lr=1e-5, head_lr=1e-3)
#     )

#     dummy = torch.randn(2, 3, 224, 224)
#     cls_logits, box_preds, centerness = model(dummy)
#     print(f"cls_logits : {cls_logits.shape}")    # [2, 256, 7]
#     print(f"box_preds  : {box_preds.shape}")     # [2, 256, 4]
#     print(f"centerness : {centerness.shape}")    # [2, 256, 1]

#     criterion = DetectionLoss()
#     targets = [
#         {"boxes": torch.tensor([[10., 20., 80., 90.]]), "labels": torch.tensor([1])},
#         {"boxes": torch.tensor([[50., 50., 150., 150.]]), "labels": torch.tensor([2])},
#     ]
#     loss, info = criterion(cls_logits, box_preds, centerness, targets)
#     print(f"Loss: {loss.item():.4f} | {info}")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import argparse
import numpy as np
import xml.etree.ElementTree as ET 
from PIL import Image 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from torchvision.ops import nms, box_iou

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from model import DetectionLoss, DINOv2Detector, DetectionHead

# ──────────────────────────────────────────────
# DATASET AND DATALOADER
# ──────────────────────────────────────────────
class NEUDataset(Dataset):
    def __init__(self, root: str, transforms=None):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.imd_dir = os.path.join(root, "IMAGES")
        self.ann_dir = os.path.join(root, "ANNOTATIONS")
        self.samples = self.__load_samples()
        
        self.class_names: dict[str, int] = {}
        count = 1
        for img_path, _ in self.samples:
            cls = self._get_class_name(img_path)
            if cls not in self.class_names:
                self.class_names[cls] = count
                count += 1

    def _get_class_name(self, path: str)-> str:
        stem  = os.path.splitext(os.path.basename(path))[0]
        return "_".join(stem.split("_")[:-1])

    def __load_samples(self):
        samples = []
        for fname in sorted(os.listdir(self.imd_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg")):
                continue
            img_path = os.path.join(self.imd_dir, fname)
            xml_name = os.path.splitext(fname)[0] + ".xml"
            xml_path = os.path.join(self.ann_dir, xml_name)
            if not os.path.exists(xml_path):
                print(f"! Annotation is not founded..")
                continue
            samples.append((img_path, xml_path))
        return samples

    def _parse_xml(self, xml_path: str):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip()
            if class_name not in self.class_names:
                continue
            bb = obj.find("bndbox")
            xmin, ymin, xmax, ymax = float(bb.find("xmin").text), float(bb.find("ymin").text), float(bb.find("xmax").text), float(bb.find("ymax").text)
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_names[class_name])
        return boxes, labels

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx:int):
        img_path, xml_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        boxes, labels = self._parse_xml(xml_path)

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "filename": os.path.basename(img_path)
        }

        if self.transforms:
            np_img = np.array(image)
            boxes_voc = target["boxes"].numpy().tolist()
            labels_np = target["labels"].numpy().tolist()
            # Albumentations dict format for bboxes
            # only process if there are valid bboxes
            try:
                transformed = self.transforms(image=np_img, bboxes=boxes_voc, labels=labels_np)
                image = transformed['image']
                
                # Check if bboxes exist after augmentation
                if len(transformed['bboxes']) > 0:
                    target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                    target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)
                else: # Edge case: object chopped off
                    target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                    target['labels'] = torch.zeros((0,), dtype=torch.int64)
            except Exception as e:
                # Fallback on errors
                image = TF.to_tensor(image)
                target['boxes'] = boxes_tensor
                target['labels'] = labels_tensor
        else:
            image = TF.to_tensor(image)

        return image, target


def get_dataloader(root, transformer, bs, split=[0.8, 0.1, 0.1]):
    dataset = NEUDataset(root=root, transforms=transformer)
    len_data = len(dataset)
    tr_data = int(len_data * split[0])
    val_data = int(len_data * split[1])
    ts_data = len_data - (tr_data + val_data)

    tr_ds, val_ds, ts_ds = random_split(dataset=dataset, lengths=[tr_data, val_data, ts_data])
    
    def collate_fn(batch):
        return tuple(zip(*batch))

    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=0)
    ts_dl = DataLoader(ts_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    return tr_dl, val_dl, ts_dl, dataset.class_names

# ──────────────────────────────────────────────
# TRAINING AND VALIDATION
# ──────────────────────────────────────────────
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, accum_steps=1):
    model.train()
    total_loss = total_cls = total_box = total_ctr = 0.0
    optimizer.zero_grad()

    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc="  Train", leave=False)
    for step, (imgs, targets) in loop:
        images = torch.stack(imgs).to(device)
        targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]

        with autocast(enabled=scaler is not None):
            cls_logits, box_preds, centerness = model(images)
            loss, info = criterion(cls_logits, box_preds, centerness, targets)
            loss = loss / accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        total_cls += info["cls_loss"]
        total_box += info["box_loss"]
        total_ctr += info["ctr_loss"]
        loop.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

    n = len(dataloader)
    return {"loss": total_loss / n, "cls_loss": total_cls / n, "box_loss": total_box / n, "ctr_loss": total_ctr / n}


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = total_cls = total_box = total_ctr = 0.0

    loop = tqdm(dataloader, desc="  Val  ", leave=False)
    for imgs, targets in loop:
        images = torch.stack(imgs).to(device)
        targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]

        with autocast(enabled=torch.cuda.is_available()):
            cls_logits, box_preds, centerness = model(images)
            loss, info = criterion(cls_logits, box_preds, centerness, targets)

        total_loss += loss.item()
        total_cls += info["cls_loss"]
        total_box += info["box_loss"]
        total_ctr += info["ctr_loss"]

    n = len(dataloader)
    return {"loss": total_loss / n, "cls_loss": total_cls / n, "box_loss": total_box / n, "ctr_loss": total_ctr / n}

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "val_loss": val_loss,
    }, path)

def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"  Checkpoint yuklandi → epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    return ckpt["epoch"]

def plot_history(history, save_path="loss_curve.png"):
    tr = [h["loss"] for h in history["train"]]
    vl = [h["loss"] for h in history["val"]]
    ep = range(1, len(tr) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(ep, tr, label="Train", marker="o")
    axes[0].plot(ep, vl, label="Val", marker="s")
    axes[0].set_title("Total Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(ep, [h["cls_loss"] for h in history["train"]], label="cls")
    axes[1].plot(ep, [h["box_loss"] for h in history["train"]], label="box")
    axes[1].plot(ep, [h["ctr_loss"] for h in history["train"]], label="ctr")
    axes[1].set_title("Train Loss components"); axes[1].legend(); axes[1].grid(True)
    axes[2].plot(ep, [h["cls_loss"] for h in history["val"]], label="cls")
    axes[2].plot(ep, [h["box_loss"] for h in history["val"]], label="box")
    axes[2].plot(ep, [h["ctr_loss"] for h in history["val"]], label="ctr")
    axes[2].set_title("Val Loss components"); axes[2].legend(); axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    
# ──────────────────────────────────────────────
# INFERENCE AND METRICS
# ──────────────────────────────────────────────
def decode_predictions(cls_logits, box_preds, img_size=224, score_thresh=0.3, nms_thresh=0.4):
    B, N, _ = cls_logits.shape
    H = W = int(N ** 0.5)
    gy, gx = torch.meshgrid(torch.arange(H, device=cls_logits.device), torch.arange(W, device=cls_logits.device), indexing="ij")
    cell_cx, cell_cy = (gx.flatten() + 0.5) / W, (gy.flatten() + 0.5) / H

    results = []
    probs = torch.softmax(cls_logits, dim=-1)
    for b in range(B):
        scores, labels = probs[b, :, 1:].max(dim=-1)
        labels = labels + 1
        mask = scores > score_thresh
        if mask.sum() == 0:
            results.append({"boxes": torch.zeros((0,4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)})
            continue
        scores, labels = scores[mask], labels[mask]
        cx, cy = box_preds[b, mask, 0] * img_size, box_preds[b, mask, 1] * img_size
        bw, bh = box_preds[b, mask, 2] * img_size, box_preds[b, mask, 3] * img_size
        xmin, ymin = (cx - bw/2).clamp(0, img_size), (cy - bh/2).clamp(0, img_size)
        xmax, ymax = (cx + bw/2).clamp(0, img_size), (cy + bh/2).clamp(0, img_size)
        boxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)

        keep_all = []
        for cls_id in labels.unique():
            cls_mask = labels == cls_id
            keep = nms(boxes[cls_mask], scores[cls_mask], nms_thresh)
            idx = cls_mask.nonzero(as_tuple=False).squeeze(1)[keep]
            keep_all.append(idx)

        if keep_all:
            keep_idx = torch.cat(keep_all)
            results.append({"boxes": boxes[keep_idx].cpu(), "scores": scores[keep_idx].cpu(), "labels": labels[keep_idx].cpu()})
        else:
            results.append({"boxes": torch.zeros((0,4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)})
    return results

def compute_ap(recalls, precisions):
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_rec = precisions[recalls >= thr]
        ap += prec_at_rec.max() if len(prec_at_rec) > 0 else 0.0
    return ap / 11

def compute_map(all_preds, all_targets, num_classes, iou_thresh=0.5):
    per_class_ap, per_class_stats = {}, {}
    for cls_id in range(1, num_classes + 1):
        tp_list, fp_list, scores_list = [], [], []
        n_gt = 0
        for pred, gt in zip(all_preds, all_targets):
            gt_boxes = gt["boxes"][gt["labels"] == cls_id]
            pred_mask = pred["labels"] == cls_id
            pred_boxes, pred_scores = pred["boxes"][pred_mask], pred["scores"][pred_mask]
            n_gt += len(gt_boxes)

            if len(pred_boxes) == 0:
                continue

            order = pred_scores.argsort(descending=True)
            pred_boxes, pred_scores = pred_boxes[order], pred_scores[order]
            matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

            for pb in pred_boxes:
                scores_list.append(pred_scores[0].item())  # Simplification context
                if len(gt_boxes) == 0:
                    tp_list.append(0); fp_list.append(1)
                    continue
                ious = box_iou(pb.unsqueeze(0), gt_boxes)[0]
                best_iou, best_idx = ious.max(0)
                if best_iou >= iou_thresh and not matched[best_idx]:
                    tp_list.append(1); fp_list.append(0)
                    matched[best_idx] = True
                else:
                    tp_list.append(0); fp_list.append(1)

        if n_gt == 0 or len(scores_list) == 0:
            per_class_ap[cls_id] = 0.0
            per_class_stats[cls_id] = {"precision": 0, "recall": 0, "f1": 0}
            continue

        tp, fp, scores_arr = np.array(tp_list), np.array(fp_list), np.array(scores_list)
        order = scores_arr.argsort()[::-1]
        tp, fp = tp[order], fp[order]
        cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
        recalls = cum_tp / (n_gt + 1e-8)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-8)
        per_class_ap[cls_id] = compute_ap(recalls, precisions)
        p, r = precisions[-1] if len(precisions) else 0, recalls[-1] if len(recalls) else 0
        per_class_stats[cls_id] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r + 1e-8),
                                    "precisions": precisions, "recalls": recalls}

    map50 = np.mean(list(per_class_ap.values())) if len(per_class_ap) > 0 else 0
    return map50, per_class_ap, per_class_stats

@torch.no_grad()
def run_inference(model, dataloader, device, score_thresh=0.3, nms_thresh=0.4, img_size=224):
    model.eval()
    all_preds, all_targets, all_images, all_filenames = [], [], [], []
    for imgs, targets in tqdm(dataloader, desc="Inference"):
        images = torch.stack(imgs).to(device)
        cls_logits, box_preds, _ = model(images) # we ignore centerness for metric decodes
        preds = decode_predictions(cls_logits, box_preds, img_size=img_size, score_thresh=score_thresh, nms_thresh=nms_thresh)
        all_preds.extend(preds)
        all_targets.extend([{"boxes": t["boxes"], "labels": t["labels"]} for t in targets])
        all_images.extend([img.cpu() for img in imgs])
        all_filenames.extend([t.get("filename", "") for t in targets])
    return all_preds, all_targets, all_images, all_filenames

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train DINOv2 Detector")
    parser.add_argument('--root', type=str, default="NEU_Dataset/NEU-DET/", help='Dataset Root')
    parser.add_argument('--epochs', type=int, default=50, help='Total Number of Epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch Size')
    parser.add_argument('--backbone-lr', type=float, default=1e-5, help='Backbone LR')
    parser.add_argument('--head-lr', type=float, default=1e-3, help='Head LR')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight Decay')
    parser.add_argument('--accum-steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--patience', type=int, default=10, help='Early Stopping Patience')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup Epochs')
    parser.add_argument('--save-dir', type=str, default="checkpoints", help='Save directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    # Use inference arg for just testing the model
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate/infer from current model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}, Epochs: {args.epochs}, Batch Size: {args.batch_size}")

    # Dataset & Dataloaders
    mean, std, im_size = [0.2250, 0.2250, 0.2250], [0.2505, 0.2505, 0.2505], 224
    
    # We must explicitly disable bboxes augmentations dropping labels 
    # Use BboxParams properly
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1, min_area=0.0))

    if not os.path.exists(args.root):
        print(f"Warning: Dataset ROOT '{args.root}' not found. Make sure data is accessible. Returning...")
        return

    tr_dl, val_dl, ts_dl, class_names = get_dataloader(root=args.root, transformer=transform, bs=args.batch_size)
    num_classes = len(class_names)
    idx_to_name = {v: k for k, v in class_names.items()}
    print(f"Classes ({num_classes}): {class_names}")

    # Model definition
    model = DINOv2Detector(num_classes=num_classes, model_name="dinov2_vits14", unfreeze_last_n=12)
    model.count_params()

    if args.eval_only:
        if args.resume:
            model.load_state_dict(torch.load(args.resume, map_location=device)["model"])
        model = model.to(device)
        print("Running Final Inference!")
        all_preds, all_targets, all_images, all_filenames = run_inference(model, ts_dl, device)
        map50, per_class_ap, per_class_stats = compute_map(all_preds, all_targets, num_classes=num_classes, iou_thresh=0.5)
        print(f"mAP@50: {map50:.4f}")
        return

    # Train setup
    os.makedirs(args.save_dir, exist_ok=True)
    model = model.to(device)
    criterion = DetectionLoss(lambda_box=5.0, lambda_ctr=1.0, focal_alpha=0.25, focal_gamma=2.0)
    optimizer = optim.AdamW(model.get_optimizer_groups(backbone_lr=args.backbone_lr, head_lr=args.head_lr, weight_decay=args.weight_decay))
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: (ep + 1) / args.warmup_epochs if ep < args.warmup_epochs else 1.0)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-7)
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(model, optimizer, cosine_scheduler, args.resume, device)

    best_val_loss = float("inf")
    patience_count = 0
    history = {"train": [], "val": []}

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        print(f"Epoch [{epoch:>3}/{args.epochs}]")
        
        train_info = train_one_epoch(model, tr_dl, criterion, optimizer, scaler, device, args.accum_steps)
        val_info = evaluate(model, val_dl, criterion, device)

        if epoch <= args.warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        history["train"].append(train_info)
        history["val"].append(val_info)
        
        print(f"  Train Loss: {train_info['loss']:.4f} | Val Loss: {val_info['loss']:.4f} | Time: {time.time()-t0:.1f}s")
        
        if val_info["loss"] < best_val_loss:
            best_val_loss = val_info["loss"]
            patience_count = 0
            save_checkpoint(model, optimizer, cosine_scheduler, epoch, best_val_loss, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  ✅ Saved Best Model (val_loss = {best_val_loss:.4f})")
        else:
            patience_count += 1
            print(f"  ⏳ No improvement ({patience_count}/{args.patience})")
            
        save_checkpoint(model, optimizer, cosine_scheduler, epoch, val_info["loss"], os.path.join(args.save_dir, "last_model.pth"))
        
        if patience_count >= args.patience:
            print(f"\n🛑 Early stopping!")
            break

    print(f"Training Finished! Best Val Loss: {best_val_loss:.4f}")
    plot_history(history, save_path=os.path.join(args.save_dir, "loss_curve.png"))


if __name__ == "__main__":
    main()

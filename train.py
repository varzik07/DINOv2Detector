import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import argparse
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from model import DetectionLoss, DINOv2Detector
from dataset import get_dataloader
from utils import save_checkpoint, load_checkpoint, plot_history

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

        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}, Epochs: {args.epochs}, Batch Size: {args.batch_size}")

    mean, std = [0.2250, 0.2250, 0.2250], [0.2505, 0.2505, 0.2505]
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1, min_area=0.0))

    if not os.path.exists(args.root):
        print(f"Dataset root '{args.root}' not found. Returning...")
        return

    tr_dl, val_dl, _, class_names = get_dataloader(root=args.root, transformer=transform, bs=args.batch_size)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    model = DINOv2Detector(num_classes=num_classes, model_name="dinov2_vits14", unfreeze_last_n=12)
    model.count_params()

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

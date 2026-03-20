import os
import argparse
import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import DINOv2Detector
from dataset import get_dataloader
from utils import decode_predictions, compute_map, plot_predictions, plot_metrics, plot_confusion_matrix

@torch.no_grad()
def run_inference(model, dataloader, device, score_thresh=0.3, nms_thresh=0.4, img_size=224):
    model.eval()
    all_preds, all_targets, all_images, all_filenames = [], [], [], []
    for imgs, targets in tqdm(dataloader, desc="Inference"):
        images = torch.stack(imgs).to(device)
        cls_logits, box_preds, _ = model(images)
        preds = decode_predictions(cls_logits, box_preds, img_size=img_size, score_thresh=score_thresh, nms_thresh=nms_thresh)
        all_preds.extend(preds)
        all_targets.extend([{"boxes": t["boxes"], "labels": t["labels"]} for t in targets])
        all_images.extend([img.cpu() for img in imgs])
        all_filenames.extend([t.get("filename", "") for t in targets])
    return all_preds, all_targets, all_images, all_filenames

def main():
    parser = argparse.ArgumentParser(description="Inference DINOv2 Detector")
    parser.add_argument('--root', type=str, default="NEU_Dataset/NEU-DET/", help='Dataset Root')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch Size')
    parser.add_argument('--weights', type=str, required=True, help='Path to checkpoint weights')
    parser.add_argument('--save-dir', type=str, default="results", help='Directory to save output plots')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}, Inferencing from {args.weights}")

    mean, std = [0.2250, 0.2250, 0.2250], [0.2505, 0.2505, 0.2505]
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1, min_area=0.0))

    if not os.path.exists(args.root):
        print(f"Dataset root '{args.root}' not found. Exiting.")
        return

    _, _, ts_dl, class_names = get_dataloader(root=args.root, transformer=transform, bs=args.batch_size)
    num_classes = len(class_names)
    idx_to_name = {v: k for k, v in class_names.items()}
    
    model = DINOv2Detector(num_classes=num_classes, model_name="dinov2_vits14", frozen=True)
    if not os.path.exists(args.weights):
        print(f"File {args.weights} not found. Exiting.")
        return
    model.load_state_dict(torch.load(args.weights, map_location=device)["model"])
    model = model.to(device)

    os.makedirs(args.save_dir, exist_ok=True)

    print("Running Inference...")
    all_preds, all_targets, all_images, all_filenames = run_inference(model, ts_dl, device)
    
    map50, per_class_ap, per_class_stats = compute_map(all_preds, all_targets, num_classes=num_classes, iou_thresh=0.5)
    
    print(f"\n{'='*40}")
    print(f"  mAP@IoU=0.5 : {map50:.4f}")
    for cls_id, ap in per_class_ap.items():
        name = idx_to_name.get(cls_id, str(cls_id))
        stats = per_class_stats.get(cls_id, {})
        print(f"  {name:<20} AP={ap:.3f} | P={stats.get('precision',0):.3f} | R={stats.get('recall',0):.3f} | F1={stats.get('f1',0):.3f}")
    print(f"{'='*40}\n")

    plot_predictions(all_images, all_preds, all_targets, all_filenames, idx_to_name, mean, std, n=12, save_path=os.path.join(args.save_dir, "predictions.png"))
    plot_metrics(per_class_ap, per_class_stats, idx_to_name, map50, save_path=os.path.join(args.save_dir, "metrics.png"))
    plot_confusion_matrix(all_preds, all_targets, idx_to_name, num_classes, save_path=os.path.join(args.save_dir, "confusion_matrix.png"))
    
if __name__ == "__main__":
    main()

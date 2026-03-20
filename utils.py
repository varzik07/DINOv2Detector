import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms, box_iou

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
            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]
            n_gt += len(gt_boxes)

            if len(pred_boxes) == 0:
                continue

            order = pred_scores.argsort(descending=True)
            pred_boxes, pred_scores = pred_boxes[order], pred_scores[order]
            matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

            for pb in pred_boxes:
                scores_list.append(pred_scores[0].item())
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
        p = precisions[-1] if len(precisions) else 0
        r = recalls[-1] if len(recalls) else 0
        per_class_stats[cls_id] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r + 1e-8),
                                    "precisions": precisions, "recalls": recalls}

    map50 = np.mean(list(per_class_ap.values())) if len(per_class_ap) > 0 else 0
    return map50, per_class_ap, per_class_stats

def denormalize(tensor, mean, std):
    t = tensor.clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)

def plot_predictions(all_images, all_preds, all_targets, all_filenames,
                     idx_to_name, mean, std, n=12, save_path="predictions.png"):
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i in range(min(n, len(all_images))):
        img = denormalize(all_images[i], mean=mean, std=std)
        np_img = img.permute(1, 2, 0).cpu().numpy()

        ax = axes[i]
        ax.imshow(np_img, cmap="gray")

        # Ground Truth (Lime)
        for box, lbl in zip(all_targets[i]["boxes"], all_targets[i]["labels"]):
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="lime", facecolor="none"))
            ax.text(x1, y1-5, f"GT:{idx_to_name.get(int(lbl), str(int(lbl)))}", color="lime", fontsize=6,
                    bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

        # Predictions (Red)
        for box, lbl, scr in zip(all_preds[i]["boxes"], all_preds[i]["labels"], all_preds[i]["scores"]):
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="red", facecolor="none", linestyle="--"))
            name = idx_to_name.get(int(lbl), str(int(lbl)))
            ax.text(x1, y2+8, f"{name}:{scr:.2f}", color="red", fontsize=6,
                    bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

        ax.set_title(all_filenames[i], fontsize=6)
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("🟩 GT    🟥 Prediction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved Predictions Plot: {save_path}")

def plot_metrics(per_class_ap, per_class_stats, idx_to_name, map50, save_path="metrics.png"):
    class_ids = sorted(per_class_ap.keys())
    names  = [idx_to_name.get(c, str(c)) for c in class_ids]
    aps    = [per_class_ap[c] for c in class_ids]
    precs  = [per_class_stats.get(c, {}).get("precision", 0) for c in class_ids]
    recs   = [per_class_stats.get(c, {}).get("recall", 0)    for c in class_ids]
    f1s    = [per_class_stats.get(c, {}).get("f1", 0)        for c in class_ids]

    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # AP per class
    bars = axes[0,0].bar(names, aps, color="steelblue", edgecolor="black")
    axes[0,0].axhline(map50, color="red", linestyle="--", label=f"mAP@50={map50:.3f}")
    axes[0,0].set_title("AP per Class (@IoU=0.5)")
    axes[0,0].set_ylim(0, 1.05)
    axes[0,0].legend(); axes[0,0].grid(axis="y", alpha=0.4)
    for bar, val in zip(bars, aps):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=8)

    # Precision / Recall / F1
    w = 0.25
    axes[0,1].bar(x - w, precs, w, label="Precision", color="royalblue")
    axes[0,1].bar(x,     recs,  w, label="Recall",    color="orange")
    axes[0,1].bar(x + w, f1s,   w, label="F1",        color="green")
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(names, rotation=15)
    axes[0,1].set_title("Precision / Recall / F1 per Class")
    axes[0,1].set_ylim(0, 1.1)
    axes[0,1].legend(); axes[0,1].grid(axis="y", alpha=0.4)

    # PR Curve
    for c in class_ids:
        if "precisions" in per_class_stats.get(c, {}):
            p = per_class_stats[c]["precisions"]
            r = per_class_stats[c]["recalls"]
            axes[1,0].plot(r, p, marker=".", label=f"{idx_to_name.get(c,c)} (AP={per_class_ap[c]:.2f})")
    axes[1,0].set_title("PR Curve")
    axes[1,0].set_xlabel("Recall"); axes[1,0].set_ylabel("Precision")
    axes[1,0].set_xlim(0, 1.05); axes[1,0].set_ylim(0, 1.05)
    axes[1,0].legend(fontsize=7); axes[1,0].grid(alpha=0.4)

    # Summary table
    axes[1,1].axis("off")
    table_data = [["Class", "AP", "Prec", "Rec", "F1"]]
    for c, name in zip(class_ids, names):
        table_data.append([
            name,
            f"{per_class_ap[c]:.3f}",
            f"{per_class_stats.get(c,{}).get('precision',0):.3f}",
            f"{per_class_stats.get(c,{}).get('recall',0):.3f}",
            f"{per_class_stats.get(c,{}).get('f1',0):.3f}",
        ])
    table_data.append(["mAP@50", f"{map50:.3f}", "-", "-", "-"])

    tbl = axes[1,1].table(cellText=table_data[1:], colLabels=table_data[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    axes[1,1].set_title("Summary Table", fontweight="bold")

    fig.suptitle(f"Inference Metrics — mAP@50: {map50:.4f}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved Metrics Plot: {save_path}")

def plot_confusion_matrix(all_preds, all_targets, idx_to_name, num_classes, iou_thresh=0.5, save_path="confusion_matrix.png"):
    n = num_classes + 1
    matrix = np.zeros((n, n), dtype=int)
    labels_list = list(range(1, num_classes + 1))

    for pred, gt in zip(all_preds, all_targets):
        gt_boxes, gt_labels = gt["boxes"], gt["labels"]
        pb, pl = pred["boxes"], pred["labels"]
        matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)

        for i, (box, lbl) in enumerate(zip(pb, pl)):
            if len(gt_boxes) == 0:
                matrix[int(lbl)][0] += 1
                continue
            ious = box_iou(box.unsqueeze(0), gt_boxes)[0]
            best_iou, best_idx = ious.max(0)
            if best_iou >= iou_thresh:
                gt_lbl = int(gt_labels[best_idx])
                matrix[int(lbl)][gt_lbl] += 1
                matched_gt[best_idx] = True
            else:
                matrix[int(lbl)][0] += 1

        for j, matched in enumerate(matched_gt):
            if not matched:
                matrix[0][int(gt_labels[j])] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_labels = ["BG"] + [idx_to_name.get(i, str(i)) for i in labels_list]
    ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels)
    ax.set_xlabel("GT"); ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="black" if matrix[i, j] < matrix.max() * 0.7 else "white", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved Confusion Matrix Plot: {save_path}")

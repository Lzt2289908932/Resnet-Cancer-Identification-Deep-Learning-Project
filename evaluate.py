import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
)
from model import get_model
from dataset import CancerCellDataset, get_test_transforms
from torch.utils.data import DataLoader

class Evaluator:
    #加载checkpoint对测试集做评估

    CLASS_NAMES = ["正常细胞 (0)", "癌细胞 (1)"]

    def __init__(self, checkpoint_path, test_dir, device=None, batch_size=32):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})

        self.model = get_model(
            num_classes=cfg.get("num_classes", 2),
            pretrained=False,
            device=self.device,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        ep = ckpt.get("epoch", "?")
        va = ckpt.get("val_acc", 0)
        print(f"loaded model (epoch {ep}, val acc {va:.2f}%)")

        test_ds = CancerCellDataset(test_dir, transform=get_test_transforms())
        self.test_loader = DataLoader(test_ds, batch_size=batch_size,
                                       shuffle=False, num_workers=4)

    @torch.no_grad()
    def predict_all(self):
        all_labels, all_preds, all_probs = [], [], []

        for images, labels in self.test_loader:
            images = images.to(self.device)
            out = self.model(images)
            probs = F.softmax(out, dim=1)
            _, predicted = out.max(1)

            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    def full_report(self, save_dir="./evaluation"):
        os.makedirs(save_dir, exist_ok=True)
        labels, preds, probs = self.predict_all()

        # classification report
        print("\n" + "=" * 50)
        report = classification_report(labels, preds,
                                        target_names=self.CLASS_NAMES, digits=4)
        print(report)

        cm = confusion_matrix(labels, preds)
        self._plot_cm(cm, save_dir)
        self._plot_roc(labels, probs, save_dir)
        self._plot_pr(labels, probs, save_dir)

        # 算一些汇总指标
        acc = np.mean(labels == preds) * 100
        sens = cm[1, 1] / cm[1].sum() * 100 if cm[1].sum() > 0 else 0
        spec = cm[0, 0] / cm[0].sum() * 100 if cm[0].sum() > 0 else 0
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val = auc(fpr, tpr)

        summary = {
            "accuracy": f"{acc:.2f}%",
            "sensitivity_recall": f"{sens:.2f}%",
            "specificity": f"{spec:.2f}%",
            "auc": f"{auc_val:.4f}",
        }
        print("summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nresults saved to {save_dir}")
        return summary

    def _plot_cm(self, cm, save_dir):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.colorbar(im)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(self.CLASS_NAMES, fontsize=11)
        ax.set_yticklabels(self.CLASS_NAMES, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=18, fontweight="bold", color=color)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

    def _plot_roc(self, labels, probs, save_dir):
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, "r-", lw=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve")
        ax.legend(fontsize=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=150)
        plt.close()

    def _plot_pr(self, labels, probs, save_dir):
        precision, recall, _ = precision_recall_curve(labels, probs)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, "b-", lw=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "pr_curve.png"), dpi=150)
        plt.close()


def plot_training_history(history_path, save_dir="./evaluation"):
    os.makedirs(save_dir, exist_ok=True)

    with open(history_path) as f:
        hist = json.load(f)

    epochs = range(1, len(hist["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, hist["train_loss"], "b-o", ms=3, label="train")
    ax1.plot(epochs, hist["val_loss"], "r-o", ms=3, label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, hist["train_acc"], "b-o", ms=3, label="train")
    ax2.plot(epochs, hist["val_acc"], "r-o", ms=3, label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"saved training curves to {save_dir}")


# 单张图片推理，用于测试
def predict_single_image(model_path, image_path, device=None):
    from PIL import Image

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    model = get_model(num_classes=2, pretrained=False, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = get_test_transforms()
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        conf, pred = probs.max(1)

    label = pred.item()
    c = conf.item()
    name = "cancer" if label == 1 else "normal"
    print(f"prediction: {name} (label={label}, conf={c:.4f})")
    return label, c, name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./checkpoints/best_model.pth")
    parser.add_argument("--test_dir", default="./data/train")
    parser.add_argument("--save_dir", default="./evaluation")
    parser.add_argument("--history", default="./checkpoints/training_history.json")
    args = parser.parse_args()

    evaluator = Evaluator(args.checkpoint, args.test_dir)
    evaluator.full_report(save_dir=args.save_dir)

    if os.path.exists(args.history):
        plot_training_history(args.history, save_dir=args.save_dir)

#经过训练后进行测试，测试目录为train的有标签目录
#准确率为95.38% 召回率95.21 AUC值为0.9878 说明此模型具有很强的癌细胞与正常细胞的区分功能
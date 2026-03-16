import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import get_model
from dataset import UnlabeledDataset, get_test_transforms

CHECKPOINT_PATH = "./checkpoints/best_model.pth"
TEST_FOLDER = r"./data/test/test_class"
OUTPUT_CSV = "./predictions.csv"
BATCH_SIZE = 32

def predict_folder(ckpt_path, folder, out_csv, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = get_model(num_classes=2, pretrained=False, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    va = ckpt.get("val_acc")
    ep = ckpt.get("epoch")
    print(f"loaded: {ckpt_path}")
    if va and ep:
        print(f"  epoch {ep}, best val acc: {va:.2f}%")

    ds = UnlabeledDataset(folder, transform=get_test_transforms())
    if len(ds) == 0:
        print("没找到图片，检查路径!")
        return
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    results = []
    n_cancer = 0
    n_normal = 0

    print("predicting...")

    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device)
            out = model(images)
            probs = F.softmax(out, dim=1)
            confs, preds = probs.max(1)

            for path, lbl, cf in zip(paths, preds.cpu().numpy(), confs.cpu().numpy()):
                lbl = int(lbl)
                cf = float(cf)
                cls = "癌细胞" if lbl == 1 else "正常细胞"
                fname = os.path.basename(path)

                results.append({
                    "filename": fname,
                    "label": lbl,
                    "class": cls,
                    "confidence": round(cf, 4),
                })

                if lbl == 1:
                    n_cancer += 1
                else:
                    n_normal += 1

                tag = "x" if lbl == 1 else "v"
                print(f"  {tag} {fname:40s} -> {cls} ({cf:.4f})")

    # 写csv
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "class", "confidence"])
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    print(f"\n{'=' * 50}")
    print(f"done! total: {total}")
    print(f"  normal: {n_normal} ({100*n_normal/total:.1f}%)")
    print(f"  cancer: {n_cancer} ({100*n_cancer/total:.1f}%)")
    print(f"saved to: {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    predict_folder(CHECKPOINT_PATH, TEST_FOLDER, OUTPUT_CSV, BATCH_SIZE)

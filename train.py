import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import json
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import get_model
from dataset import create_dataloaders_with_split


class Trainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.best_acc = 0.0
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        os.makedirs(config["save_dir"], exist_ok=True)

        self.model = get_model(
            num_classes=config["num_classes"],
            pretrained=config["pretrained"],
            device=self.device,
        )

        # loss
        if config.get("class_weights"):
            w = torch.tensor(config["class_weights"], dtype=torch.float).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=w)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # optimizer
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if config["optimizer"] == "adam":
            self.optimizer = Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])
        # lr scheduler - val loss不降就砍半
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=5, factor=0.5)#盯loss的耐心

        self.train_loader, self.val_loader = create_dataloaders_with_split(
            train_dir=config["train_dir"],
            val_ratio=config["val_ratio"],
            batch_size=config["batch_size"],
            img_size=config["img_size"],
            num_workers=config["num_workers"],
        )

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(images)
            loss = self.criterion(out, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f"  batch {i+1}/{len(self.train_loader)} "
                      f"loss={loss.item():.4f} acc={100.*correct/total:.1f}%")

        return total_loss / total, 100.0 * correct / total

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            out = self.model(images)
            loss = self.criterion(out, labels)

            total_loss += loss.item() * images.size(0)
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        return total_loss / total, 100.0 * correct / total

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "config": self.config,
        }
        torch.save(ckpt, os.path.join(self.config["save_dir"], "last_checkpoint.pth"))

        if is_best:
            torch.save(ckpt, os.path.join(self.config["save_dir"], "best_model.pth"))
            print(f"  ** new best: {val_acc:.2f}%")

    def train(self):
        print("=" * 50)
        print(f"device: {self.device}")
        print(f"epochs: {self.config['epochs']}, bs: {self.config['batch_size']}, "
              f"lr: {self.config['lr']}")
        print("=" * 50)

        patience_cnt = 0
        t0 = time.time()

        for epoch in range(1, self.config["epochs"] + 1):
            t_epoch = time.time()

            # 之前试过两阶段训练（先冻结再解冻），效果不如直接全部训练
            # if epoch == self.config.get("unfreeze_epoch", 999):
            #     print(f"\n>> epoch {epoch}: unfreezing backbone")
            #     self.model.unfreeze_backbone(unfreeze_from="layer3")
            #     for pg in self.optimizer.param_groups:
            #         pg["lr"] = self.config["lr"] * 0.1

            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.evaluate()

            self.scheduler.step(val_loss)

            # 记录history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            dt = time.time() - t_epoch
            lr_now = self.optimizer.param_groups[0]["lr"]
            print(f"\nepoch {epoch}/{self.config['epochs']} ({dt:.1f}s) lr={lr_now:.6f}")
            print(f"  train: loss={train_loss:.4f} acc={train_acc:.2f}%")
            print(f"  val:   loss={val_loss:.4f} acc={val_acc:.2f}%")

            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                patience_cnt = 0
            else:
                patience_cnt += 1

            self.save_checkpoint(epoch, val_acc, is_best)

            # early stopping
            if patience_cnt >= self.config["patience"]:
                print(f"\nearly stopping at epoch {epoch} "
                      f"(no improvement for {self.config['patience']} epochs)")
                break

        elapsed = time.time() - t0
        print(f"\ndone. {elapsed/60:.1f} min, best val acc: {self.best_acc:.2f}%")

        # save training history
        hist_path = os.path.join(self.config["save_dir"], "training_history.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

DEFAULT_CONFIG = {
    "train_dir": "./data/train",
    "val_ratio": 0.2,
    "img_size": 224,
    "batch_size": 32,
    "num_workers": 0,
    "num_classes": 2,
    "pretrained": True,
    # "freeze_backbone": True,    # 试过，不如直接全部训练
    # "unfreeze_epoch": 15,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "adam",
    "patience": 10, #盯acc的耐心值
    "class_weights": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "./checkpoints",
}


if __name__ == "__main__":
    config = DEFAULT_CONFIG.copy()
    # 跑了几组实验之后发现这组参数效果比较好
    #冻结实验下在第25个epoch发生早停 best acc 86%
    #非冻结试验下在51个epoch发生早停 best acc 93.12
    #训练集acc 93.25% 测试集acc 93.12%
    #确定基本不存在过拟合现象
    config["train_dir"] = r"./data/train"
    config["batch_size"] = 32
    config["epochs"] = 60
    config["lr"] = 5e-4
    config["patience"] = 15
    config["weight_decay"] = 1e-3      # 防过拟合
    # config["class_weights"] = [1.0, 1.5]  # 试过加权，效果一般

    trainer = Trainer(config)
    trainer.train()

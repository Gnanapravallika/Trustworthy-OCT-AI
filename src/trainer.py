"""Trainer loop module for training and validating TrustOCT models."""

import os
import sys
import time
from typing import Dict, Optional, Tuple
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.losses import EdlLoss
from src.models import coral_loss


class Trainer:
    """Rigorous training and validation engine for TrustOCT models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_config_path: str,
        model_config_path: str,
        experiment_dir: str
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_dir = experiment_dir

        self.train_cfg = self._load_yaml(train_config_path)
        self.model_cfg = self._load_yaml(model_config_path)

        # Set device
        device_str = self.train_cfg.get("device", "cuda")
        self.device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
        self.model = self.model.to(self.device)

        # Build optimizer
        opt_type = self.train_cfg.get("optimizer", "adamw").lower()
        lr = self.train_cfg.get("learning_rate", 1e-4)
        wd = self.train_cfg.get("weight_decay", 1e-4)
        
        if opt_type == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

        # Build scheduler
        self.epochs = self.train_cfg.get("epochs", 30)
        sched_type = self.train_cfg.get("scheduler", "cosine").lower()
        if sched_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_cfg.get("cosine_T_max", self.epochs)
            )
        else:
            self.scheduler = None

        # Build primary loss function
        self.head_type = self.model_cfg.get("head", "edl").lower()
        self.num_classes = self.model_cfg.get("num_classes", 4)
        
        if self.head_type == "edl":
            edl_cfg = self.train_cfg.get("edl", {})
            self.criterion = EdlLoss(
                num_classes=self.num_classes,
                annealing_epochs=edl_cfg.get("annealing_epochs", 10)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None

        log_cfg = self.train_cfg.get("logging", {})
        self.tb_writer = None
        if log_cfg.get("tensorboard", True):
            tb_dir = os.path.join(experiment_dir, "tb_logs")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.patience = self.train_cfg.get("checkpoint", {}).get("patience", 7)

    def _load_yaml(self, path: str) -> dict:
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    if self.head_type == "edl":
                        _, alpha = outputs
                        loss = self.criterion(alpha, targets, epoch)
                        preds = torch.argmax(alpha, dim=1)
                    else:
                        loss = self.criterion(outputs, targets)
                        preds = torch.argmax(outputs, dim=1)

                    if self.model_cfg.get("domain_generalization", "identity").lower() == "coral":
                        if images.size(0) >= 4:
                            half = images.size(0) // 2
                            last_feats = self.model.dg.last_features
                            c_loss = coral_loss(last_feats[:half], last_feats[half:])
                            loss += self.train_cfg.get("coral", {}).get("weight", 0.5) * c_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if self.head_type == "edl":
                    _, alpha = outputs
                    loss = self.criterion(alpha, targets, epoch)
                    preds = torch.argmax(alpha, dim=1)
                else:
                    loss = self.criterion(outputs, targets)
                    preds = torch.argmax(outputs, dim=1)

                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (preds == targets).sum().item()
            total += images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                if self.head_type == "edl":
                    _, alpha = outputs
                    loss = self.criterion(alpha, targets, epoch)
                    preds = torch.argmax(alpha, dim=1)
                else:
                    loss = self.criterion(outputs, targets)
                    preds = torch.argmax(outputs, dim=1)

                running_loss += loss.item() * images.size(0)
                correct += (preds == targets).sum().item()
                total += images.size(0)

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self) -> None:
        print(f"Starting training on device: {self.device}")
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            if self.scheduler:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {lr:.6f}")

            if self.tb_writer:
                self.tb_writer.add_scalar("Loss/train", train_loss, epoch)
                self.tb_writer.add_scalar("Loss/val", val_loss, epoch)
                self.tb_writer.add_scalar("Accuracy/train", train_acc, epoch)
                self.tb_writer.add_scalar("Accuracy/val", val_acc, epoch)
                self.tb_writer.add_scalar("LR", lr, epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1} due to validation loss plateau.")
                    break

            if (epoch + 1) % self.train_cfg.get("checkpoint", {}).get("save_freq", 5) == 0:
                self._save_checkpoint(epoch, val_loss, is_best=False)

        if self.tb_writer:
            self.tb_writer.close()
        print("Training complete.")

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        os.makedirs(self.experiment_dir, exist_ok=True)
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_loss": self.best_val_loss,
            "val_loss": val_loss
        }

        if is_best:
            filepath = os.path.join(self.experiment_dir, "weights_best.pth")
        else:
            filepath = os.path.join(self.experiment_dir, f"checkpoint_epoch_{epoch+1}.pth")

        torch.save(state, filepath)
        print(f"Saved weights: {os.path.basename(filepath)}")

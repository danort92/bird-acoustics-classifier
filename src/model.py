"""
src/model.py — EfficientNet-B0 fine-tuning for bird acoustic classification.

Loads mel-spectrogram PNGs produced by src/preprocessing.py, fine-tunes a
pretrained EfficientNet-B0 with a stratified train/val/test split, tracks
every experiment with MLflow, and saves the best checkpoint to models/.

Quick start:
    from src.model import BirdTrainer, TrainingConfig

    cfg     = TrainingConfig.from_yaml()          # reads config/default.yaml
    trainer = BirdTrainer(cfg)
    best    = trainer.train()                      # returns Path to best_model.pt
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All tunable parameters for the training pipeline."""

    processed_dir:    str   = "data/processed"
    output_dir:       str   = "models"
    model_name:       str   = "efficientnet_b0"
    epochs:           int   = 30
    batch_size:       int   = 32
    learning_rate:    float = 1e-3
    val_split:        float = 0.15
    test_split:       float = 0.15
    seed:             int   = 42
    num_workers:      int   = 4
    img_size:         Tuple[int, int] = field(default_factory=lambda: (224, 224))
    use_scheduler:    bool  = True
    patience:         int   = 7       # early-stopping patience (epochs)
    experiment_name:  str   = "bird-acoustics-classifier"
    tracking_uri:     str   = "mlruns"

    @classmethod
    def from_yaml(cls, path: str = "config/default.yaml") -> "TrainingConfig":
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
        t = cfg.get("training", {})
        m = cfg.get("mlflow",   {})
        a = cfg.get("audio",    {})
        return cls(
            processed_dir   = cfg["data"]["processed_dir"],
            output_dir      = t.get("output_dir",    "models"),
            model_name      = t.get("model",          "efficientnet_b0"),
            epochs          = t.get("epochs",         30),
            batch_size      = t.get("batch_size",     32),
            learning_rate   = t.get("learning_rate",  1e-3),
            val_split       = t.get("val_split",      0.15),
            test_split      = t.get("test_split",     0.15),
            seed            = t.get("seed",           42),
            num_workers     = t.get("num_workers",    4),
            img_size        = tuple(a.get("img_size", [224, 224])),
            experiment_name = m.get("experiment_name", "bird-acoustics-classifier"),
            tracking_uri    = m.get("tracking_uri",    "mlruns"),
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BirdDataset(Dataset):
    """
    Loads mel-spectrogram PNGs from::

        <processed_dir>/<species>/*.png

    Each species subdirectory becomes one class; classes are sorted
    alphabetically so the mapping is deterministic across runs.
    """

    def __init__(
        self,
        processed_dir: str,
        transform=None,
        species: Optional[List[str]] = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.transform = transform

        all_dirs = sorted(
            d for d in self.processed_dir.iterdir()
            if d.is_dir() and list(d.glob("*.png"))
        )
        if species:
            allowed = {s.replace(" ", "_").lower() for s in species}
            all_dirs = [d for d in all_dirs if d.name.lower() in allowed]

        self.classes: List[str] = [d.name for d in all_dirs]
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[Path, int]] = []
        for d in all_dirs:
            idx = self.class_to_idx[d.name]
            for p in sorted(d.glob("*.png")):
                self.samples.append((p, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def num_classes(self) -> int:
        return len(self.classes)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(
    img_size: Tuple[int, int] = (224, 224),
    augment: bool = True,
):
    """Returns (train_transform, val_transform)."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose(
        [
            transforms.Resize(img_size),
            *(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                ]
                if augment else []
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 with a custom head for *num_classes* outputs."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model   = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BirdTrainer:
    """Manages dataset splitting, training loop, MLflow logging, and checkpointing."""

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self._set_seed(cfg.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self.classes: List[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _run_epoch(
        self,
        loader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple[float, float]:
        """One training or evaluation pass. Returns (avg_loss, accuracy)."""
        training = optimizer is not None
        model.train() if training else model.eval()
        total_loss = correct = total = 0

        with torch.set_grad_enabled(training):
            for imgs, labels in tqdm(loader, leave=False, desc="train" if training else "eval"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * imgs.size(0)
                correct    += (logits.argmax(1) == labels).sum().item()
                total      += imgs.size(0)

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_dataloaders(
        self,
        species: Optional[List[str]] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """Stratified train / val / test split. Returns (train, val, test, num_classes)."""
        cfg = self.cfg
        train_tf, val_tf = get_transforms(cfg.img_size, augment=True)

        # Reference dataset to get the full index/label list
        ref_ds  = BirdDataset(cfg.processed_dir, transform=None, species=species)
        n       = len(ref_ds)
        indices = list(range(n))
        labels  = [ref_ds.samples[i][1] for i in indices]

        self.classes = ref_ds.classes
        num_classes  = ref_ds.num_classes

        # Stratified split: train+val vs test
        val_size_adjusted = cfg.val_split / (1.0 - cfg.test_split)
        train_idx, test_idx = train_test_split(
            indices, test_size=cfg.test_split, stratify=labels, random_state=cfg.seed
        )
        train_labels = [labels[i] for i in train_idx]
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_size_adjusted,
            stratify=train_labels, random_state=cfg.seed,
        )

        logger.info(
            f"Dataset: {n} images | {num_classes} classes | "
            f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}"
        )

        def _subset(idx: List[int], tf) -> Subset:
            ds = BirdDataset(cfg.processed_dir, transform=tf, species=species)
            return Subset(ds, idx)

        kw = dict(
            batch_size  = cfg.batch_size,
            num_workers = cfg.num_workers,
            pin_memory  = self.device.type == "cuda",
        )
        return (
            DataLoader(_subset(train_idx, train_tf), shuffle=True,  **kw),
            DataLoader(_subset(val_idx,   val_tf),   shuffle=False, **kw),
            DataLoader(_subset(test_idx,  val_tf),   shuffle=False, **kw),
            num_classes,
        )

    def train(
        self,
        species: Optional[List[str]] = None,
    ) -> Tuple[Path, Dict]:
        """
        Run the full training loop.

        Returns
        -------
        best_path : Path
            Location of the best checkpoint (``models/best_model.pt``).
        history : dict
            Per-epoch metrics for plotting::

                {
                  "train_loss": [...], "train_acc": [...],
                  "val_loss":   [...], "val_acc":   [...],
                  "lr":         [...],
                }
        """
        cfg = self.cfg
        train_loader, val_loader, test_loader, num_classes = self.build_dataloaders(species)

        model     = build_model(num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
        scheduler = (
            CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
            if cfg.use_scheduler else None
        )

        best_val_loss = float("inf")
        patience_cnt  = 0
        best_path     = Path(cfg.output_dir) / "best_model.pt"
        history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "lr":         [],
        }

        mlflow.set_tracking_uri(cfg.tracking_uri)
        mlflow.set_experiment(cfg.experiment_name)

        with mlflow.start_run():
            mlflow.log_params({
                "model":         cfg.model_name,
                "epochs":        cfg.epochs,
                "batch_size":    cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "val_split":     cfg.val_split,
                "test_split":    cfg.test_split,
                "seed":          cfg.seed,
                "num_classes":   num_classes,
                "img_size":      f"{cfg.img_size[0]}x{cfg.img_size[1]}",
                "device":        str(self.device),
                "scheduler":     "cosine" if cfg.use_scheduler else "none",
            })

            for epoch in range(1, cfg.epochs + 1):
                train_loss, train_acc = self._run_epoch(train_loader, model, criterion, optimizer)
                val_loss,   val_acc   = self._run_epoch(val_loader,   model, criterion)
                lr = optimizer.param_groups[0]["lr"]

                if scheduler:
                    scheduler.step()

                # Log to MLflow
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_acc":  train_acc,
                        "val_loss":   val_loss,
                        "val_acc":    val_acc,
                        "lr":         lr,
                    },
                    step=epoch,
                )

                # Append to history
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["lr"].append(lr)

                logger.info(
                    f"Epoch {epoch:3d}/{cfg.epochs} | "
                    f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
                    f"val loss={val_loss:.4f} acc={val_acc:.3f} | "
                    f"lr={lr:.2e}"
                )

                # Checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_cnt  = 0
                    torch.save(
                        {
                            "epoch":                epoch,
                            "model_state_dict":     model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss":             val_loss,
                            "val_acc":              val_acc,
                            "classes":              self.classes,
                            "num_classes":          num_classes,
                            "img_size":             cfg.img_size,
                            "model_name":           cfg.model_name,
                        },
                        best_path,
                    )
                    logger.info(f"  ✓ Best val_loss={best_val_loss:.4f} — saved to {best_path}")
                else:
                    patience_cnt += 1
                    if patience_cnt >= cfg.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break

            # Final test evaluation
            test_loss, test_acc = self._run_epoch(test_loader, model, criterion)
            mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
            mlflow.log_artifact(str(best_path))
            logger.info(f"Test  | loss={test_loss:.4f}  acc={test_acc:.3f}")
            history["test_loss"] = test_loss
            history["test_acc"]  = test_acc

        return best_path, history

    def evaluate(
        self,
        checkpoint_path: str | Path,
        species: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a checkpoint and run inference on the test split.

        Returns
        -------
        y_true, y_pred : np.ndarray of shape (N,)
        """
        _, _, test_loader, num_classes = self.build_dataloaders(species)
        ckpt  = torch.load(checkpoint_path, map_location=self.device)
        model = build_model(num_classes, pretrained=False).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        all_true, all_pred = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc="Evaluating"):
                imgs = imgs.to(self.device)
                preds = model(imgs).argmax(1).cpu().numpy()
                all_true.extend(labels.numpy())
                all_pred.extend(preds)

        return np.array(all_true), np.array(all_pred)


# ---------------------------------------------------------------------------
# Inference helper (for deployment / app)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str | Path, device: str = "cpu") -> Tuple[nn.Module, List[str], Tuple[int, int]]:
    """
    Load a saved checkpoint for inference.

    Returns
    -------
    model     : nn.Module (eval mode)
    classes   : list of class names
    img_size  : (W, H) tuple
    """
    ckpt       = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt["num_classes"]
    model       = build_model(num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["classes"], ckpt["img_size"]


def predict(
    image_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cpu",
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Predict the species for a single spectrogram image.

    Returns a list of (class_name, probability) tuples, sorted by probability.
    """
    model, classes, img_size = load_model(checkpoint_path, device)
    _, val_tf = get_transforms(img_size, augment=False)

    img    = Image.open(image_path).convert("RGB")
    tensor = val_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(classes[i], float(probs[i])) for i in top_idx]

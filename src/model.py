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
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def _safe_num_workers(n: int) -> int:
    """Return 0 inside Jupyter/Colab to avoid multiprocessing teardown errors."""
    try:
        get_ipython  # type: ignore[name-defined]
        return 0
    except NameError:
        return n


# ---------------------------------------------------------------------------
# Custom spectrogram transforms (applied on tensors after ToTensor)
# ---------------------------------------------------------------------------

class FrequencyMasking:
    """Zero out *num_masks* random horizontal bands of up to *max_mask_param* rows.

    Simulates recordings where certain frequency ranges are masked by noise
    or outside the microphone's sensitivity — forces the model to classify
    with partial frequency information.
    """

    def __init__(self, max_mask_param: int = 20, num_masks: int = 2):
        self.max_mask_param = max_mask_param
        self.num_masks = num_masks

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        _, H, _ = tensor.shape
        tensor = tensor.clone()
        for _ in range(self.num_masks):
            f = random.randint(1, self.max_mask_param)
            f0 = random.randint(0, max(0, H - f))
            tensor[:, f0:f0 + f, :] = 0.0
        return tensor


class TimeMasking:
    """Zero out *num_masks* random vertical bands of up to *max_mask_param* columns.

    Simulates recordings with brief silences, wind bursts, or clipping —
    forces the model to classify even when part of the call is missing.
    """

    def __init__(self, max_mask_param: int = 40, num_masks: int = 2):
        self.max_mask_param = max_mask_param
        self.num_masks = num_masks

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        _, _, W = tensor.shape
        tensor = tensor.clone()
        for _ in range(self.num_masks):
            t = random.randint(1, self.max_mask_param)
            t0 = random.randint(0, max(0, W - t))
            tensor[:, :, t0:t0 + t] = 0.0
        return tensor


class GaussianNoise:
    """Add zero-mean Gaussian noise — simulates background noise in field recordings."""

    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All tunable parameters for the training pipeline."""

    processed_dir:        str   = "data/processed"
    output_dir:           str   = "models"
    checkpoint_name:      str   = "best_model.pt"
    model_name:           str   = "efficientnet_b0"
    epochs:               int   = 30
    batch_size:           int   = 32
    learning_rate:        float = 1e-3
    val_split:            float = 0.15
    test_split:           float = 0.15
    seed:                 int   = 42
    num_workers:          int   = 4
    img_size:             Tuple[int, int] = field(default_factory=lambda: (224, 224))
    use_scheduler:        bool  = True
    patience:             int   = 7
    # Augmentation strategy: 'none' | 'basic' | 'specaugment'
    augment_strategy:     str   = "specaugment"
    # Label smoothing: 0.0 = off, 0.1 recommended to reduce overconfidence
    label_smoothing:      float = 0.1
    # WeightedRandomSampler to balance rare species during training
    use_weighted_sampler: bool  = True
    # Progressive unfreezing: train head only → unfreeze backbone at unfreeze_epoch
    progressive_unfreeze: bool  = True
    unfreeze_epoch:       int   = 5
    experiment_name:      str   = "bird-acoustics-classifier"
    tracking_uri:         str   = "mlruns"

    @classmethod
    def from_yaml(cls, path: str = "config/default.yaml") -> "TrainingConfig":
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
        t = cfg.get("training", {})
        m = cfg.get("mlflow",   {})
        a = cfg.get("audio",    {})
        return cls(
            processed_dir        = cfg["data"]["processed_dir"],
            output_dir           = t.get("output_dir",           "models"),
            checkpoint_name      = t.get("checkpoint_name",      "best_model.pt"),
            model_name           = t.get("model",                "efficientnet_b0"),
            epochs               = t.get("epochs",               30),
            batch_size           = t.get("batch_size",           32),
            learning_rate        = t.get("learning_rate",        1e-3),
            val_split            = t.get("val_split",            0.15),
            test_split           = t.get("test_split",           0.15),
            seed                 = t.get("seed",                 42),
            num_workers          = t.get("num_workers",          4),
            img_size             = tuple(a.get("img_size",       [224, 224])),
            augment_strategy     = t.get("augment_strategy",     "specaugment"),
            label_smoothing      = t.get("label_smoothing",      0.1),
            use_weighted_sampler = t.get("use_weighted_sampler", True),
            progressive_unfreeze = t.get("progressive_unfreeze", True),
            unfreeze_epoch       = t.get("unfreeze_epoch",       5),
            experiment_name      = m.get("experiment_name",      "bird-acoustics-classifier"),
            tracking_uri         = m.get("tracking_uri",         "mlruns"),
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
    augment_strategy: str = "specaugment",
):
    """Return (train_transform, val_transform).

    augment_strategy
    ----------------
    ``'none'``
        No augmentation — pure baseline / ablation.
    ``'basic'``
        Original approach: RandomHorizontalFlip + ColorJitter.
        Kept for comparison; not recommended for spectrograms.
    ``'specaugment'``
        Domain-appropriate pipeline:

        * FrequencyMasking — masks random frequency bands
        * TimeMasking      — masks random time segments
        * GaussianNoise    — simulates field recording noise
        * RandomErasing    — drops random rectangular patches
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    base      = [transforms.Resize(img_size)]
    to_tensor = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    if augment_strategy == "basic":
        pil_aug    = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        tensor_aug = []
    elif augment_strategy == "specaugment":
        pil_aug    = []
        tensor_aug = [
            FrequencyMasking(max_mask_param=20, num_masks=2),
            TimeMasking(max_mask_param=40, num_masks=2),
            GaussianNoise(std=0.02),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ]
    else:  # 'none'
        pil_aug    = []
        tensor_aug = []

    train_tf = transforms.Compose(base + pil_aug + to_tensor + tensor_aug)
    val_tf   = transforms.Compose(base + to_tensor)
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
        train_tf, val_tf = get_transforms(cfg.img_size, augment_strategy=cfg.augment_strategy)

        ref_ds  = BirdDataset(cfg.processed_dir, transform=None, species=species)
        n       = len(ref_ds)
        indices = list(range(n))
        labels  = [ref_ds.samples[i][1] for i in indices]

        self.classes = ref_ds.classes
        num_classes  = ref_ds.num_classes

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
            # Re-use ref_ds.samples so all three subsets share the same file list
            # and indices.  Creating a fresh BirdDataset here risks a different
            # file count if the directory is modified between scans, which makes
            # the global indices (0..N-1) overflow the new dataset's samples list.
            ds = BirdDataset.__new__(BirdDataset)
            ds.processed_dir = ref_ds.processed_dir
            ds.classes       = ref_ds.classes
            ds.class_to_idx  = ref_ds.class_to_idx
            ds.samples       = ref_ds.samples
            ds.transform     = tf
            return Subset(ds, idx)

        nw = _safe_num_workers(cfg.num_workers)
        kw = dict(
            batch_size         = cfg.batch_size,
            num_workers        = nw,
            pin_memory         = self.device.type == "cuda",
            persistent_workers = nw > 0,
        )

        # WeightedRandomSampler: over-sample rare species during training
        if cfg.use_weighted_sampler:
            class_counts  = np.bincount(train_labels, minlength=num_classes).astype(float)
            class_weights = 1.0 / np.maximum(class_counts, 1.0)
            sample_w = torch.tensor([class_weights[l] for l in train_labels], dtype=torch.float)
            sampler  = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
            train_loader = DataLoader(_subset(train_idx, train_tf), sampler=sampler, **kw)
        else:
            train_loader = DataLoader(_subset(train_idx, train_tf), shuffle=True, **kw)

        return (
            train_loader,
            DataLoader(_subset(val_idx,  val_tf), shuffle=False, **kw),
            DataLoader(_subset(test_idx, val_tf), shuffle=False, **kw),
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
            Location of the best checkpoint.
        history : dict
            Per-epoch metrics::

                {
                  "train_loss": [...], "train_acc": [...],
                  "val_loss":   [...], "val_acc":   [...],
                  "lr":         [...],
                  "test_loss":  float, "test_acc":  float,
                }
        """
        cfg = self.cfg
        train_loader, val_loader, test_loader, num_classes = self.build_dataloaders(species)

        model     = build_model(num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        # Progressive unfreezing: freeze backbone, train only the head first
        if cfg.progressive_unfreeze:
            for param in model.features.parameters():
                param.requires_grad = False
            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.learning_rate,
            )
            logger.info(
                "Progressive unfreeze ON — backbone frozen for first %d epochs", cfg.unfreeze_epoch
            )
        else:
            optimizer = Adam(model.parameters(), lr=cfg.learning_rate)

        scheduler = (
            CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
            if cfg.use_scheduler else None
        )

        best_val_loss = float("inf")
        patience_cnt  = 0
        best_path     = Path(cfg.output_dir) / cfg.checkpoint_name
        history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "lr":         [],
        }

        mlflow.set_tracking_uri(cfg.tracking_uri)
        mlflow.set_experiment(cfg.experiment_name)

        _HDR = (
            f"{'Epoch':>8} │ {'Train Loss':>10} {'Train Acc':>9} │ "
            f"{'Val Loss':>8} {'Val Acc':>7} │ {'LR':>9}"
        )
        _SEP = "─" * len(_HDR)

        with mlflow.start_run():
            mlflow.log_params({
                "model":                cfg.model_name,
                "epochs":               cfg.epochs,
                "batch_size":           cfg.batch_size,
                "learning_rate":        cfg.learning_rate,
                "val_split":            cfg.val_split,
                "test_split":           cfg.test_split,
                "seed":                 cfg.seed,
                "num_classes":          num_classes,
                "img_size":             f"{cfg.img_size[0]}x{cfg.img_size[1]}",
                "device":               str(self.device),
                "scheduler":            "cosine" if cfg.use_scheduler else "none",
                "augment_strategy":     cfg.augment_strategy,
                "label_smoothing":      cfg.label_smoothing,
                "use_weighted_sampler": cfg.use_weighted_sampler,
                "progressive_unfreeze": cfg.progressive_unfreeze,
                "unfreeze_epoch":       cfg.unfreeze_epoch,
            })

            print(_SEP)
            print(_HDR)
            print(_SEP)

            for epoch in range(1, cfg.epochs + 1):

                # Unfreeze backbone after warm-up epochs
                if cfg.progressive_unfreeze and epoch == cfg.unfreeze_epoch + 1:
                    for param in model.features.parameters():
                        param.requires_grad = True
                    optimizer.add_param_group({
                        "params": list(model.features.parameters()),
                        "lr":     cfg.learning_rate * 0.1,
                    })
                    logger.info(
                        "Backbone unfrozen at epoch %d — backbone lr=%.2e",
                        epoch, cfg.learning_rate * 0.1,
                    )

                train_loss, train_acc = self._run_epoch(train_loader, model, criterion, optimizer)
                val_loss,   val_acc   = self._run_epoch(val_loader,   model, criterion)
                lr = optimizer.param_groups[0]["lr"]

                if scheduler:
                    scheduler.step()

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

                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["lr"].append(lr)

                checkpoint_marker = "  "
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_cnt  = 0
                    checkpoint_marker = " *"
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
                else:
                    patience_cnt += 1

                unfreeze_tag = (
                    " [unfreeze]"
                    if cfg.progressive_unfreeze and epoch == cfg.unfreeze_epoch + 1
                    else ""
                )
                print(
                    f"{epoch:>4}/{cfg.epochs:<4} │ "
                    f"{train_loss:>10.4f} {train_acc:>9.3f} │ "
                    f"{val_loss:>8.4f} {val_acc:>7.3f} │ "
                    f"{lr:>9.2e}"
                    f"{checkpoint_marker}{unfreeze_tag}"
                )

                if patience_cnt >= cfg.patience:
                    print(_SEP)
                    print(f"  Early stopping at epoch {epoch} (patience={cfg.patience})")
                    break

            print(_SEP)

            test_loss, test_acc = self._run_epoch(test_loader, model, criterion)
            mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
            mlflow.log_artifact(str(best_path))
            print(f"  Test  │ loss={test_loss:.4f}  acc={test_acc:.3f}")
            print(_SEP)
            history["test_loss"] = test_loss
            history["test_acc"]  = test_acc

            # Patch the best checkpoint to include the full history so the
            # file is self-contained and no sidecar JSON is needed.
            ckpt_data = torch.load(best_path, map_location="cpu", weights_only=False)
            ckpt_data["history"] = history
            torch.save(ckpt_data, best_path)

        return best_path, history

    def _restore_from_checkpoint(self, checkpoint_path: Path) -> Dict:
        """Populate ``self.classes`` and return history from a saved checkpoint.

        Used by :meth:`load_or_train` to skip training when a checkpoint
        already exists.  The checkpoint must have been written by
        :meth:`train` (which embeds history at the end of the run).
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.classes = ckpt["classes"]
        return ckpt.get("history", {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "lr":         [],
            "test_loss":  0.0, "test_acc": 0.0,
        })

    @classmethod
    def load_or_train(
        cls,
        cfg: "TrainingConfig",
        species: Optional[List[str]] = None,
        force: bool = False,
    ) -> Tuple["BirdTrainer", Path, Dict]:
        """Load an existing checkpoint or run training if none is found.

        Parameters
        ----------
        cfg :
            Training configuration.
        species :
            Optional species filter forwarded to :meth:`train`.
        force :
            When *True*, always retrain even if a checkpoint exists.

        Returns
        -------
        trainer  : BirdTrainer (``classes`` already populated)
        path     : Path to the best checkpoint
        history  : metrics dict (same structure as :meth:`train`)
        """
        ckpt_path = Path(cfg.output_dir) / cfg.checkpoint_name
        trainer   = cls(cfg)
        if not force and ckpt_path.exists():
            history = trainer._restore_from_checkpoint(ckpt_path)
            logger.info("Checkpoint loaded (skipping training): %s", ckpt_path)
            return trainer, ckpt_path, history
        path, history = trainer.train(species=species)
        return trainer, path, history

    def evaluate(
        self,
        checkpoint_path: str | Path,
        species: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load a checkpoint and run inference on the test split.

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

def load_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> Tuple[nn.Module, List[str], Tuple[int, int]]:
    """Load a saved checkpoint for inference.

    Returns
    -------
    model     : nn.Module (eval mode)
    classes   : list of class names
    img_size  : (W, H) tuple
    """
    ckpt        = torch.load(checkpoint_path, map_location=device)
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
    """Predict the species for a single spectrogram image.

    Returns a list of (class_name, probability) tuples, sorted by probability.
    """
    model, classes, img_size = load_model(checkpoint_path, device)
    _, val_tf = get_transforms(img_size, augment_strategy="none")

    img    = Image.open(image_path).convert("RGB")
    tensor = val_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(classes[i], float(probs[i])) for i in top_idx]

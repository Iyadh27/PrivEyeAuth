import argparse
import glob
import os
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from emmixformer_fixed import EmMixformer


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER) for biometric verification.
    
    Args:
        labels: True labels (0 for genuine, 1 for impostor)
        scores: Similarity scores (higher = more similar)
    
    Returns:
        EER value (between 0 and 1)
    """
    try:
        # Remove any NaN or Inf values
        valid_mask = ~(np.isnan(scores) | np.isinf(scores) | np.isnan(labels) | np.isinf(labels))
        if valid_mask.sum() < 10:  # Need at least 10 valid samples
            return 0.5
        
        scores = scores[valid_mask]
        labels = labels[valid_mask]
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # EER is where FPR = 1 - TPR (FNR)
        fnr = 1 - tpr
        
        # Find threshold where FPR = FNR
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
        
        return float(eer)
    except Exception as e:
        print(f"Warning: EER computation failed: {e}")
        return 0.5


class CTBUEyeMovementDataset(Dataset):
    """
    Dataset for CTBU-EMglasses eye movement .xlsx files from Tobii Pro Glasses 3.
    
    The data format matches the EMglasses dataset described in the EmMixformer paper
    (Section III-A), collected at 50 Hz sampling rate without head stabilization.
    """

    def __init__(
        self,
        data_dir: str,
        x_col: str = "Gaze point X",  # Updated to match actual column names
        y_col: str = "Gaze point Y",  # Updated to match actual column names
        min_len: int = 50,  # At 50Hz, this is 1 second
        max_len: int = 2048,
        normalize: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.x_col = x_col
        self.y_col = y_col
        self.min_len = min_len
        self.max_len = max_len
        self.normalize = normalize

        pattern = os.path.join(data_dir, "*.xlsx")
        self.files: List[str] = sorted(glob.glob(pattern))
        if not self.files:
            raise RuntimeError(f"No .xlsx files found in {data_dir}")

        # Extract subject IDs from filenames
        subject_ids: List[str] = []
        self.file_subject: List[str] = []
        
        for f in self.files:
            base = os.path.basename(f)
            subj = None
            
            # Try multiple extraction strategies
            if '_' in base:
                parts = base.split('_')
                if parts[0]:
                    subj = parts[0]
            
            if subj is None and '-' in base:
                parts = base.split('-')
                if parts[0]:
                    subj = parts[0]
            
            if subj is None:
                match = re.search(r'\d+', base)
                if match:
                    subj = match.group(0)
            
            if subj is None:
                subj = base[:3]
            
            subject_ids.append(subj)
            self.file_subject.append(subj)
        
        # Print first 3 examples for verification
        print("Dataset initialization - First 3 files and extracted IDs:")
        for i in range(min(3, len(self.files))):
            print(f"  File: {os.path.basename(self.files[i])} -> Subject ID: {self.file_subject[i]}")
        
        unique_ids = sorted(set(subject_ids))
        self.subj_to_label: Dict[str, int] = {sid: i for i, sid in enumerate(unique_ids)}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.files[idx]
        subj = self.file_subject[idx]
        label = self.subj_to_label[subj]

        try:
            df = pd.read_excel(path)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            # Return zero sequence if file is corrupted
            seq = np.zeros((self.min_len, 2), dtype=np.float32)
            return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)

        # Try to find gaze columns with different naming conventions
        x_col_actual = None
        y_col_actual = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'gaze' in col_lower and 'point' in col_lower and 'x' in col_lower:
                x_col_actual = col
            if 'gaze' in col_lower and 'point' in col_lower and 'y' in col_lower:
                y_col_actual = col
        
        # Fallback to original column names
        if x_col_actual is None:
            x_col_actual = self.x_col
        if y_col_actual is None:
            y_col_actual = self.y_col

        # Check if columns exist
        if x_col_actual not in df.columns or y_col_actual not in df.columns:
            print(f"Warning: Columns not found in {path}. Available columns: {list(df.columns)[:5]}...")
            seq = np.zeros((self.min_len, 2), dtype=np.float32)
            return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)

        # Extract gaze coordinates and drop NaN
        df_clean = df[[x_col_actual, y_col_actual]].dropna()
        
        if len(df_clean) == 0:
            # All NaN - return zero sequence
            seq = np.zeros((self.min_len, 2), dtype=np.float32)
        else:
            x_vals = df_clean[x_col_actual].to_numpy(dtype=np.float32)
            y_vals = df_clean[y_col_actual].to_numpy(dtype=np.float32)
            
            # Stack into sequence
            seq = np.stack([x_vals, y_vals], axis=-1)  # (T, 2)

        # Handle very short sequences
        if seq.shape[0] < self.min_len:
            # Pad by repeating last frame
            pad_len = self.min_len - seq.shape[0]
            if seq.shape[0] > 0:
                last = seq[-1:]
                seq = np.concatenate([seq, np.repeat(last, pad_len, axis=0)], axis=0)
            else:
                seq = np.zeros((self.min_len, 2), dtype=np.float32)

        # Crop very long sequences
        if seq.shape[0] > self.max_len:
            seq = seq[:self.max_len, :]

        # Replace any remaining NaN/Inf
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize to prevent extreme values
        if self.normalize and seq.shape[0] > 0:
            # Per-sequence normalization
            mean = seq.mean(axis=0, keepdims=True)
            std = seq.std(axis=0, keepdims=True) + 1e-8
            seq = (seq - mean) / std
            
            # Clip to reasonable range
            seq = np.clip(seq, -10, 10)

        x = torch.from_numpy(seq)  # (T, 2)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def pad_collate(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function that pads variable-length sequences to the max length in the batch.
    """
    sequences, labels = zip(*batch)
    lengths = [seq.size(0) for seq in sequences]
    max_len = max(lengths)

    batch_size = len(sequences)
    x_padded = torch.zeros(batch_size, max_len, 2, dtype=sequences[0].dtype)

    for i, seq in enumerate(sequences):
        T = seq.size(0)
        x_padded[i, :T, :] = seq

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return x_padded, labels_tensor


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """
    Train for one epoch with gradient clipping to prevent NaN.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)  # (B, T, 2)
        y = y.to(device)
        
        # Check for NaN in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: NaN/Inf in input data at batch {batch_idx}. Skipping.")
            continue

        optimizer.zero_grad()
        
        try:
            logits = model(x)  # (B, num_classes)
            
            # Check for NaN in output
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Warning: NaN/Inf in model output at batch {batch_idx}. Skipping.")
                continue
            
            loss = criterion(logits, y)
            
            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss at batch {batch_idx}. Skipping.")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(loader)}: loss={loss.item():.4f}")
                
        except RuntimeError as e:
            print(f"Warning: Runtime error at batch {batch_idx}: {e}. Skipping.")
            continue

    if total_samples == 0:
        print("Warning: No valid samples processed in this epoch!")
        return float('nan'), 0.0

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    print(f"Epoch {epoch}: train loss={avg_loss:.4f}, acc={accuracy:.4f}")
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "val",
) -> Tuple[float, float, float]:
    """
    Evaluate the model and compute EER.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_probs = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        
        # Check for NaN in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: NaN/Inf in {split_name} input at batch {batch_idx}. Skipping.")
            continue
        
        try:
            logits = model(x)
            
            # Check for NaN in output
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Warning: NaN/Inf in {split_name} output at batch {batch_idx}. Skipping.")
                continue
            
            loss = criterion(logits, y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss in {split_name} at batch {batch_idx}. Skipping.")
                continue
            
            probs = torch.softmax(logits, dim=1)  # (B, num_classes)
            
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print(f"Warning: NaN/Inf in probabilities for {split_name}. Skipping.")
                continue

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
            
            # Store for EER computation
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
        except RuntimeError as e:
            print(f"Warning: Runtime error in {split_name} at batch {batch_idx}: {e}. Skipping.")
            continue

    if total_samples == 0:
        print(f"Warning: No valid samples in {split_name}!")
        return float('nan'), 0.0, 0.5

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    # Compute EER
    if len(all_probs) == 0:
        eer = 0.5
    else:
        all_probs = np.concatenate(all_probs, axis=0)  # (N, num_classes)
        all_labels = np.concatenate(all_labels, axis=0)  # (N,)
        
        # Compute genuine and impostor scores
        genuine_scores = []
        impostor_scores = []
        
        for i in range(len(all_labels)):
            true_label = all_labels[i]
            prob_vec = all_probs[i]
            
            # Genuine score: probability of correct class
            genuine_scores.append(float(prob_vec[true_label]))
            
            # Impostor score: max probability of any incorrect class
            mask = np.arange(len(prob_vec)) != true_label
            if mask.sum() > 0:
                impostor_scores.append(float(prob_vec[mask].max()))
        
        if len(genuine_scores) > 0 and len(impostor_scores) > 0:
            scores = np.array(genuine_scores + impostor_scores)
            labels = np.concatenate([
                np.zeros(len(genuine_scores)),
                np.ones(len(impostor_scores))
            ])
            eer = compute_eer(labels, scores)
        else:
            eer = 0.5
    
    print(f"{split_name}: loss={avg_loss:.4f}, acc={accuracy:.4f}, EER={eer:.4f}")
    return avg_loss, accuracy, eer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EmMixformer on eye movement dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to directory containing .xlsx files",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)  # Lower learning rate
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument(
        "--velocity-threshold",
        type=float,
        default=40.0,
        help="Velocity threshold (deg/s) for splitting slow/fast streams",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="emmixformer_best.pth",
        help="Where to save the best model",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and splits
    dataset = CTBUEyeMovementDataset(
        data_dir=args.data_dir,
        min_len=args.min_len,
        max_len=args.max_len,
        normalize=True,
    )

    num_classes = len(dataset.subj_to_label)
    print(f"Found {len(dataset)} sequences from {num_classes} subjects.")

    # Split data
    n_total = len(dataset)
    n_test = int(n_total * args.test_split)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val - n_test
    
    train_ds, val_ds, test_ds = random_split(
        dataset, 
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Split: train={n_train}, val={n_val}, test={n_test}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Model
    model = EmMixformer(
        num_classes=num_classes,
        velocity_threshold=args.velocity_threshold,
        cnn_base_channels=32,
        mix_att_hidden_dim=128,
        mix_trans_dim=128,
        mix_fourier_dim=128,
        transformer_heads=4,
        transformer_layers=2,
        transformer_ff_dim=256,
        dropout=0.1,
        num_mix_blocks=2,
    ).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5,
        # verbose=True
    )

    best_val_eer = 1.0  # Lower is better for EER
    best_state = None

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.max_grad_norm
        )
        
        if np.isnan(train_loss):
            print("Training loss is NaN! Stopping training.")
            break
        
        val_loss, val_acc, val_eer = evaluate(model, val_loader, device, split_name="val")
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Save best model based on EER (lower is better)
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_eer": val_eer,
                "train_acc": train_acc,
                "args": vars(args),
            }
            torch.save(best_state, args.save_path)
            print(f"✓ Saved new best model with val EER={val_eer:.4f} to {args.save_path}")

    # Load best model and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
        print(f"\nLoaded best model from epoch {best_state['epoch']} for testing.")
        print(f"Best validation EER: {best_state['val_eer']:.4f}")

    print("\n" + "="*60)
    print("Final Test Evaluation:")
    print("="*60)
    test_loss, test_acc, test_eer = evaluate(model, test_loader, device, split_name="test")
    
    print(f"\nTraining complete!")
    print(f"Best validation EER: {best_val_eer:.4f}")
    print(f"Test EER: {test_eer:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()

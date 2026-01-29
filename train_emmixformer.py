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

from emmixformer import EmMixformer


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER) for biometric verification.
    
    Args:
        labels: True labels (0 for genuine, 1 for impostor)
        scores: Similarity scores (higher = more similar)
    
    Returns:
        EER value (between 0 and 1)
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # EER is where FPR = 1 - TPR (FNR)
    fnr = 1 - tpr
    
    # Find threshold where FPR = FNR
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    
    # EER value
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    
    return float(eer)


class CTBUEyeMovementDataset(Dataset):
    """
    Dataset for CTBU-EMglasses Dynamic .xlsx files.

    Assumptions (adjust as needed):
      - Each .xlsx file corresponds to one trial / sequence.
      - Subject ID is encoded in the filename as the first 3 characters, e.g. '001-D-1.xlsx'.
      - Each file contains columns for gaze coordinates named 'x' and 'y'
        (or 'X' and 'Y'); this can be customized via arguments.
    """

    def __init__(
        self,
        data_dir: str,
        x_col: str = "x",
        y_col: str = "y",
        min_len: int = 10,
        max_len: int = 2048,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.x_col = x_col
        self.y_col = y_col
        self.min_len = min_len
        self.max_len = max_len

        pattern = os.path.join(data_dir, "*.xlsx")
        self.files: List[str] = sorted(glob.glob(pattern))
        if not self.files:
            raise RuntimeError(f"No .xlsx files found in {data_dir}")

        # Build subject-id to label mapping from filenames
        # Robust extraction: try multiple patterns
        subject_ids: List[str] = []
        self.file_subject: List[str] = []
        
        for f in self.files:
            base = os.path.basename(f)
            # Try multiple extraction strategies
            subj = None
            
            # Strategy 1: Split by underscore and take first part
            if '_' in base:
                parts = base.split('_')
                if parts[0]:
                    subj = parts[0]
            
            # Strategy 2: Split by hyphen and take first part
            if subj is None and '-' in base:
                parts = base.split('-')
                if parts[0]:
                    subj = parts[0]
            
            # Strategy 3: Extract first numeric sequence
            if subj is None:
                match = re.search(r'\d+', base)
                if match:
                    subj = match.group(0)
            
            # Strategy 4: Fallback to first 3 characters
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

        df = pd.read_excel(path)

        # Drop rows with NaN in gaze coordinates if the expected columns exist
        subset_cols = [c for c in [self.x_col, self.y_col] if c in df.columns]
        if subset_cols:
            df = df.dropna(subset=subset_cols)

        # Try lowercase then uppercase column names if needed
        if self.x_col not in df.columns or self.y_col not in df.columns:
            alt_x = self.x_col.upper()
            alt_y = self.y_col.upper()
            if alt_x in df.columns and alt_y in df.columns:
                # Drop rows with NaN in gaze coordinates
                df = df.dropna(subset=[alt_x, alt_y])
                x_vals = df[alt_x].to_numpy(dtype=np.float32)
                y_vals = df[alt_y].to_numpy(dtype=np.float32)
            else:
                raise KeyError(
                    f"Columns '{self.x_col},{self.y_col}' or '{alt_x},{alt_y}' "
                    f"not found in file {path}. Available columns: {list(df.columns)}"
                )
        else:
            x_vals = df[self.x_col].to_numpy(dtype=np.float32)
            y_vals = df[self.y_col].to_numpy(dtype=np.float32)

        # Handle empty dataframe or very short sequences after dropping NaNs
        if len(x_vals) == 0 or len(y_vals) == 0:
            # Return zero sequence if dataframe is empty
            seq = np.zeros((self.min_len, 2), dtype=np.float32)
        else:
            seq = np.stack([x_vals, y_vals], axis=-1)  # (T, 2)

        # Filter very short sequences
        if seq.shape[0] < self.min_len:
            # Simple fallback: repeat last frame to reach min_len
            pad_len = self.min_len - seq.shape[0]
            last = seq[-1:]
            seq = np.concatenate([seq, np.repeat(last, pad_len, axis=0)], axis=0)

        # Optionally crop very long sequences
        if seq.shape[0] > self.max_len:
            seq = seq[: self.max_len, :]

        # Replace any NaN/Inf in sequence with zeros
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clamp extreme values to prevent numerical instability
        seq = np.clip(seq, -1e6, 1e6)

        x = torch.from_numpy(seq)  # (T, 2)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def pad_collate(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function that pads variable-length sequences to the max length in the batch.

    Input batch: list of (seq, label) where seq is (T, 2)
    Output:
      x_padded: (B, T_max, 2)
      labels:   (B,)
    """
    sequences, labels = zip(*batch)  # sequences: list[(T_i, 2)]
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
) -> Tuple[float, float]:
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
            print(f"Warning: NaN/Inf detected in input batch {batch_idx}. Skipping batch.")
            continue

        optimizer.zero_grad()
        logits = model(x)  # (B, num_classes)
        
        # Check for NaN in logits before computing loss
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: NaN/Inf detected in model output batch {batch_idx}. Skipping batch.")
            continue
        
        loss = criterion(logits, y)
        
        # Skip backward pass if loss is NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected in batch {batch_idx}. Skipping backward pass.")
            continue
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f"Epoch {epoch}: train loss={avg_loss:.4f}, acc={accuracy:.4f}")
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "val",
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()
    
    # For EER computation: collect all genuine and impostor scores
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            # Check for NaN in input
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: NaN/Inf detected in {split_name} input. Skipping batch.")
                continue

            logits = model(x)
            
            # Check for NaN in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Warning: NaN/Inf detected in {split_name} model output. Skipping batch.")
                continue
            
            loss = criterion(logits, y)
            
            # Skip if loss is NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected in {split_name}. Skipping batch.")
                continue
            
            # Get probabilities via softmax
            probs = torch.softmax(logits, dim=1)  # (B, num_classes)
            
            # Replace any NaN/Inf in probabilities with uniform distribution
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print(f"Warning: NaN/Inf in probabilities for {split_name}. Replacing with uniform.")
                probs = torch.ones_like(probs) / probs.size(1)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
            
            # Store probabilities and labels for EER (only if valid)
            probs_np = probs.cpu().numpy()
            if not (np.isnan(probs_np).any() or np.isinf(probs_np).any()):
                all_probs.append(probs_np)
                all_labels.append(y.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    # Compute EER
    if len(all_probs) == 0:
        print(f"Warning: No valid probabilities collected for {split_name}. Setting EER=0.5.")
        eer = 0.5
    else:
        all_probs = np.concatenate(all_probs, axis=0)  # (N, num_classes)
        all_labels = np.concatenate(all_labels, axis=0)  # (N,)
        
        # Filter out any rows with NaN/Inf in probabilities
        valid_mask = ~(np.isnan(all_probs).any(axis=1) | np.isinf(all_probs).any(axis=1))
        if valid_mask.sum() == 0:
            print(f"Warning: No valid probability rows for {split_name}. Setting EER=0.5.")
            eer = 0.5
        else:
            all_probs = all_probs[valid_mask]
            all_labels = all_labels[valid_mask]
            
            # For each sample:
            # - Genuine score = probability assigned to correct class
            # - Impostor score = max probability assigned to any other class
            genuine_scores = []
            impostor_scores = []
            
            for i in range(len(all_labels)):
                true_label = all_labels[i]
                prob_vec = all_probs[i]
                
                # Skip if this probability vector has NaN/Inf
                if np.isnan(prob_vec).any() or np.isinf(prob_vec).any():
                    continue
                
                # Genuine score: probability of correct class
                genuine_scores.append(float(prob_vec[true_label]))
                
                # Impostor score: max probability of any incorrect class
                mask = np.arange(len(prob_vec)) != true_label
                if mask.sum() > 0:
                    impostor_max = float(prob_vec[mask].max())
                    if not (np.isnan(impostor_max) or np.isinf(impostor_max)):
                        impostor_scores.append(impostor_max)
            
            # Create binary labels: 0 for genuine, 1 for impostor
            if len(genuine_scores) == 0 and len(impostor_scores) == 0:
                print(f"Warning: No valid scores for EER computation in {split_name}. Setting EER=0.5.")
                eer = 0.5
            else:
                scores = np.array(genuine_scores + impostor_scores)
                labels = np.concatenate([
                    np.zeros(len(genuine_scores)),
                    np.ones(len(impostor_scores))
                ])
                
                # Final safety check before calling sklearn
                if scores.size == 0 or np.isnan(scores).any() or np.isinf(scores).any():
                    print(f"Warning: EER scores contain NaN/Inf for split '{split_name}'. Setting EER=0.5.")
                    eer = 0.5
                else:
                    try:
                        eer = compute_eer(labels, scores)
                        if np.isnan(eer) or np.isinf(eer):
                            print(f"Warning: EER computation returned NaN/Inf for {split_name}. Setting EER=0.5.")
                            eer = 0.5
                    except Exception as e:
                        print(f"Warning: EER computation failed for {split_name}: {e}. Setting EER=0.5.")
                        eer = 0.5
    
    print(f"{split_name} loss={avg_loss:.4f}, acc={accuracy:.4f}, EER={eer:.4f}")
    return avg_loss, accuracy, eer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EmMixformer on CTBU-EMglasses Dynamic dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to directory containing Dynamic .xlsx files",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--x-col", type=str, default="x", help="Column name for x coordinate")
    parser.add_argument("--y-col", type=str, default="y", help="Column name for y coordinate")
    parser.add_argument("--min-len", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument(
        "--velocity-threshold",
        type=float,
        default=40.0,
        help="Velocity threshold for splitting slow/fast streams",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="emmixformer_ctbu.pth",
        help="Where to save the best model",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and splits
    dataset = CTBUEyeMovementDataset(
        data_dir=args.data_dir,
        x_col=args.x_col,
        y_col=args.y_col,
        min_len=args.min_len,
        max_len=args.max_len,
    )

    num_classes = len(dataset.subj_to_label)
    print(f"Found {len(dataset)} sequences from {num_classes} subjects.")

    n_total = len(dataset)
    n_test = int(n_total * args.test_split)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val - n_test
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate,
    )

    # Model
    model = EmMixformer(
        num_classes=num_classes,
        velocity_threshold=args.velocity_threshold,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch)
        _, val_acc, val_eer = evaluate(model, val_loader, device, split_name="val")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_eer": val_eer,
                "args": vars(args),
            }
            torch.save(best_state, args.save_path)
            print(f"Saved new best model with val acc={val_acc:.4f}, val EER={val_eer:.4f} to {args.save_path}")

    # Load best model and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
        print(f"Loaded best model from epoch {best_state['epoch']} for testing.")

    evaluate(model, test_loader, device, split_name="test")


if __name__ == "__main__":
    main()


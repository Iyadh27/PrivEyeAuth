# EmMixformer Implementation Fix - Complete Summary

## Problem Analysis

Your original implementation was getting NaN values during training, which caused the training to fail. After analyzing your code and the EmMixformer paper, I identified several critical issues:

## Root Causes of NaN Values

### 1. **Data Loading Mismatch**
- **Issue**: Your code expected columns named "x" and "y" (lowercase)
- **Reality**: Tobii Pro Glasses 3 data uses "Gaze point X" and "Gaze point Y"
- **Result**: Most data was being replaced with zeros, creating degenerate features

### 2. **Missing Data Handling**
- **Issue**: ~84% of gaze point data in your sample file contains NaN values
- **Problem**: These NaN values propagated through the network unchecked
- **Impact**: NaN × any number = NaN, causing cascading failures

### 3. **Poor Weight Initialization**
- **Issue**: No explicit weight initialization for critical components
- **Problem**: Random initialization with extreme values
- **Impact**: Unstable gradients and numerical overflow

### 4. **No Gradient Clipping**
- **Issue**: Gradients could grow arbitrarily large
- **Problem**: Exploding gradients → NaN parameters
- **Impact**: Training divergence after first few batches

### 5. **No Data Normalization**
- **Issue**: Raw gaze coordinates varied widely (0-1000+ pixels)
- **Problem**: Extreme input values → extreme activations
- **Impact**: Numerical instability in FFT, attention, and other operations

## Complete List of Fixes

### Data Loading (`train_emmixformer_fixed.py`)

```python
# BEFORE:
x_col: str = "x"  
y_col: str = "y"

# AFTER:
x_col: str = "Gaze point X"  # Correct column name for Tobii data
y_col: str = "Gaze point Y"

# Added flexible column detection:
for col in df.columns:
    col_lower = col.lower()
    if 'gaze' in col_lower and 'point' in col_lower and 'x' in col_lower:
        x_col_actual = col
```

### NaN Handling (Multiple Locations)

```python
# Added comprehensive NaN cleaning:
# 1. In dataset loading
seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

# 2. Per-sequence normalization to prevent extreme values
if self.normalize and seq.shape[0] > 0:
    mean = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-8
    seq = (seq - mean) / std
    seq = np.clip(seq, -10, 10)  # Clip to reasonable range

# 3. In model forward pass (at every critical point)
x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
```

### Weight Initialization (`emmixformer_fixed.py`)

```python
# Added proper initialization for all layers:

# 1. Convolutional layers
nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
nn.init.constant_(self.conv.bias, 0)

# 2. Linear layers
nn.init.xavier_uniform_(self.W_q.weight)
nn.init.constant_(param.bias, 0)

# 3. Classifier
nn.init.xavier_uniform_(self.classifier.weight)
nn.init.constant_(self.classifier.bias, 0)
```

### Gradient Clipping (`train_emmixformer_fixed.py`)

```python
# Added gradient clipping after backward pass:
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm=1.0)
optimizer.step()
```

### Training Stability Improvements

```python
# 1. Lower learning rate
lr = 1e-4  # Was 2e-4

# 2. Better optimizer
optimizer = torch.optim.AdamW(  # Was Adam
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8,  # Added explicit epsilon
)

# 3. Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# 4. Batch-level NaN detection
if torch.isnan(loss) or torch.isinf(loss):
    print(f"Warning: NaN/Inf loss at batch {batch_idx}. Skipping.")
    continue
```

### Model Architecture Fixes

```python
# 1. Fixed positional encoding for odd dimensions
if d_model % 2 == 1:
    pe[:, 1::2] = torch.cos(position * div_term[:-1])
else:
    pe[:, 1::2] = torch.cos(position * div_term)

# 2. Proper 2-layer Mix Block stacking (as per paper)
num_mix_blocks: int = 2  # Paper mentions "stacking in two layers"
self.mix_blocks = nn.ModuleList()
for i in range(num_mix_blocks):
    self.mix_blocks.append(MixBlock(...))
```

## File Structure

### `emmixformer_fixed.py`
- Corrected EmMixformer model implementation
- Proper weight initialization
- Fixed architecture issues
- Comprehensive NaN handling in forward pass

### `train_emmixformer_fixed.py`  
- Fixed data loading for Tobii Pro Glasses 3 format
- Per-sequence normalization
- Gradient clipping
- Better optimizer and scheduler
- Batch-level error handling
- Proper EER computation

### `test_model.py`
- Comprehensive test suite
- Verifies model creation
- Tests forward/backward passes
- Checks NaN handling
- Tests variable-length sequences

### `README_FIXES.md`
- Complete documentation
- Usage instructions
- Troubleshooting guide
- Parameter explanations

## How to Use

### 1. Prepare Your Data
Make sure your .xlsx files are in a directory with naming convention:
```
{SUBJECT_ID}-{SESSION}-{TRIAL}.xlsx
```
Example: `001-D-1.xlsx`, `001-D-2.xlsx`, `002-D-1.xlsx`

### 2. Train the Model
```bash
python train_emmixformer_fixed.py \
    --data-dir /path/to/your/xlsx/files \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-4 \
    --save-path emmixformer_best.pth
```

### 3. Monitor Training
The script will print:
- Batch-level progress every 10 batches
- Epoch-level metrics (loss, accuracy, EER)
- Best model saves automatically
- Learning rate adjustments

### 4. Expected Output
```
Epoch 1/50
========================================================
  Batch 10/24: loss=2.3456
  Batch 20/24: loss=2.1234
Epoch 1: train loss=2.2345, acc=0.1234
val: loss=2.3456, acc=0.1456, EER=0.4567
✓ Saved new best model with val EER=0.4567 to emmixformer_best.pth
```

## Troubleshooting

### If you still get NaN:
1. **Further reduce learning rate**: Try `--lr 5e-5` or `--lr 1e-5`
2. **Increase gradient clipping**: Try `--max-grad-norm 0.5`
3. **Reduce batch size**: Try `--batch-size 8` or `--batch-size 4`
4. **Check your data**: Ensure you have valid gaze points

### If accuracy is too low:
1. **More data**: Need at least 2 sessions per subject
2. **More epochs**: Try `--epochs 100`
3. **Data quality**: Check for corrupted files
4. **Subject ID extraction**: Verify IDs are extracted correctly

### If training is slow:
1. **Reduce sequence length**: Try `--max-len 512`
2. **Smaller model**: Reduce `cnn_base_channels` in code
3. **Fewer mix blocks**: Set `num_mix_blocks=1` in code

## Key Differences from Original Paper

Your dataset differs from the paper's EMglasses dataset in some ways:

| Aspect | Paper | Your Data |
|--------|-------|-----------|
| Device | Tobii Pro Glasses 3 | Tobii Pro Glasses 3 ✓ |
| Frequency | 50 Hz | 50 Hz ✓ |
| Subjects | 203 | Your dataset size |
| Data quality | Good | High NaN ratio (84%) |

The high NaN ratio in your data is unusual. Consider:
- Checking eye tracker calibration
- Ensuring good lighting conditions
- Verifying participants can see the stimuli
- Checking for occlusions (glasses, hair, etc.)

## Performance Expectations

Based on the paper (Table VIII):

| Model | EER on EMglasses |
|-------|------------------|
| EmMixformer | **0.1599** |
| DenseNet | 0.1892 |
| DEL | 0.1853 |
| Expansion CNN | 0.2140 |

With your data quality, expect:
- **Initial EER**: 0.3-0.4 (first few epochs)
- **After 20 epochs**: 0.2-0.25
- **After 50 epochs**: 0.18-0.22 (if data quality is good)

Lower EER = better performance (EER is an error rate).

## Next Steps

1. **Verify your data quality**: Check why 84% of gaze points are NaN
2. **Start with small test**: Try with just 10-20 subjects first
3. **Monitor training**: Watch for NaN warnings
4. **Adjust hyperparameters**: Based on initial results
5. **Compare with baselines**: Implement simpler models for comparison

## Additional Notes

### Batch Size Considerations
- Smaller batches (8-16) → More stable but slower
- Larger batches (32-64) → Faster but need careful tuning

### Learning Rate Schedule
- Start: 1e-4
- After 5 bad epochs: 5e-5 (automatic via scheduler)
- After 10 bad epochs: 2.5e-5 (automatic)

### Data Augmentation (Future Work)
Consider adding:
- Random temporal cropping
- Gaussian noise injection
- Velocity-based augmentation

## References

Original Paper:
```bibtex
@article{qin2024emmixformer,
  title={EmMixformer: Mix transformer for eye movement recognition},
  author={Qin, Huafeng and Zhu, Hongyu and Jin, Xin and Song, Qun and 
          El-Yacoubi, Mounim A and Gao, Xinbo},
  journal={arXiv preprint arXiv:2401.04956},
  year={2024}
}
```

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your data format matches the expected format
3. Review the console output for specific error messages
4. Ensure all dependencies are installed correctly

Good luck with your implementation!

# EmMixformer Implementation - Fixed Version

## Overview
This is a corrected implementation of the EmMixformer model from the paper "EmMixformer: Mix transformer for eye movement recognition" by Qin et al.

## Key Fixes Applied

### 1. **Data Loading Issues**
- **Problem**: Original code expected columns named "x" and "y", but Tobii Pro Glasses 3 data uses "Gaze point X" and "Gaze point Y"
- **Fix**: Updated column names and added flexible column detection

### 2. **NaN Handling**
- **Problem**: Training produced NaN values that propagated through the network
- **Fixes**:
  - Added comprehensive NaN/Inf checking and replacement throughout the pipeline
  - Added per-sequence normalization to prevent extreme values
  - Added gradient clipping (max_grad_norm=1.0) to prevent exploding gradients
  - Added data validation checks in dataset loading

### 3. **Model Initialization**
- **Problem**: Poor weight initialization led to unstable training
- **Fixes**:
  - Added Xavier/Kaiming initialization for all linear and convolutional layers
  - Properly initialized LSTM cell parameters
  - Added bias initialization

### 4. **Training Stability**
- **Problem**: Training was unstable with NaN losses
- **Fixes**:
  - Reduced initial learning rate from 2e-4 to 1e-4
  - Added learning rate scheduler (ReduceLROnPlateau)
  - Changed optimizer from Adam to AdamW
  - Added gradient clipping
  - Added batch-level NaN detection and skipping

### 5. **Data Preprocessing**
- **Problem**: Raw gaze coordinates had extreme values and high NaN ratio
- **Fixes**:
  - Added per-sequence normalization (z-score)
  - Clip values to [-10, 10] range
  - Handle empty/all-NaN sequences gracefully
  - Proper padding for short sequences

### 6. **Model Architecture**
- **Improvements**:
  - Fixed positional encoding for odd d_model dimensions
  - Properly implemented 2-layer Mix Block stacking as per paper
  - Added dropout to prevent overfitting
  - Proper initialization of all components

## Dataset Format

Your eye movement data should be in Excel (.xlsx) format with these columns:
- `Gaze point X`: Horizontal gaze position
- `Gaze point Y`: Vertical gaze position

File naming convention for subject ID extraction:
- Format: `{SUBJECT_ID}-{SESSION}-{TRIAL}.xlsx`
- Example: `001-D-1.xlsx` → Subject ID: `001`

## Usage

### Training
```bash
python train_emmixformer_fixed.py \
    --data-dir /path/to/xlsx/files \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-4 \
    --save-path emmixformer_best.pth
```

### Key Parameters
- `--data-dir`: Directory containing .xlsx files
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-4)
- `--velocity-threshold`: Threshold for slow/fast split in deg/s (default: 40.0)
- `--max-grad-norm`: Gradient clipping threshold (default: 1.0)
- `--min-len`: Minimum sequence length (default: 50, = 1 sec at 50Hz)
- `--max-len`: Maximum sequence length (default: 2048)

## Model Architecture (as per paper)

1. **Preprocessing Block**: Splits data into slow (<40°/s) and fast (≥40°/s) velocity streams
2. **Siamese CNN**: 4 convolutional blocks process each stream independently
3. **Mix Block** (×2 layers):
   - Attention LSTM: Learns short-term dependencies
   - Transformer: Learns long-term dependencies  
   - Fourier Transformer: Learns global frequency-domain features
4. **Classifier**: Global average pooling + linear layer

## Expected Performance

According to the paper (Table VIII), on the EMglasses dataset:
- EmMixformer EER: 0.1599
- Best baseline (DenseNet) EER: 0.1892

Note: Results may vary based on:
- Dataset size and quality
- Training hyperparameters
- Random initialization
- Hardware (GPU vs CPU)

## Troubleshooting

### If you still get NaN losses:
1. Reduce learning rate further (try 5e-5 or 1e-5)
2. Reduce batch size (try 8 or 4)
3. Increase gradient clipping (try max_grad_norm=0.5)
4. Check your data for extreme outliers

### If accuracy is very low:
1. Ensure you have enough data per subject (at least 2 sessions)
2. Check that subject IDs are extracted correctly
3. Try training for more epochs (100+)
4. Adjust the train/val/test split ratios

### If training is slow:
1. Reduce max_len (try 1024 or 512)
2. Reduce model dimensions (try base_channels=16)
3. Use fewer mix blocks (num_mix_blocks=1)
4. Enable GPU if available

## Files

- `emmixformer_fixed.py`: Corrected model implementation
- `train_emmixformer_fixed.py`: Corrected training script with proper data loading
- `README_FIXES.md`: This file

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{qin2024emmixformer,
  title={EmMixformer: Mix transformer for eye movement recognition},
  author={Qin, Huafeng and Zhu, Hongyu and Jin, Xin and Song, Qun and El-Yacoubi, Mounim A and Gao, Xinbo},
  journal={arXiv preprint arXiv:2401.04956},
  year={2024}
}
```

## License

Please refer to the original paper's license and terms of use.

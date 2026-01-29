#!/usr/bin/env python3
"""
Test script to verify EmMixformer model works correctly
"""

import torch
import numpy as np
from emmixformer_fixed import EmMixformer

def test_model_creation():
    """Test that model can be created"""
    print("Testing model creation...")
    model = EmMixformer(num_classes=10)
    print("✓ Model created successfully")
    return model

def test_forward_pass(model):
    """Test forward pass with random data"""
    print("\nTesting forward pass...")
    
    # Create random input
    batch_size = 4
    seq_len = 512
    x = torch.randn(batch_size, seq_len, 2)
    
    # Forward pass
    logits = model(x)
    
    # Check output shape
    assert logits.shape == (batch_size, 10), f"Expected shape (4, 10), got {logits.shape}"
    
    # Check no NaN or Inf
    assert not torch.isnan(logits).any(), "Output contains NaN!"
    assert not torch.isinf(logits).any(), "Output contains Inf!"
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    return logits

def test_backward_pass(model):
    """Test backward pass"""
    print("\nTesting backward pass...")
    
    # Create random input and target
    x = torch.randn(4, 512, 2)
    target = torch.randint(0, 10, (4,))
    
    # Forward pass
    logits = model(x)
    
    # Compute loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, target)
    
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}!"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}!"
    
    assert has_grad, "No gradients computed!"
    
    print("✓ Backward pass successful")
    print("✓ Gradients are valid (no NaN/Inf)")

def test_variable_length():
    """Test with variable-length sequences"""
    print("\nTesting variable-length sequences...")
    
    model = EmMixformer(num_classes=5)
    
    # Test different sequence lengths
    seq_lengths = [100, 256, 512, 1024]
    
    for seq_len in seq_lengths:
        x = torch.randn(2, seq_len, 2)
        logits = model(x)
        assert logits.shape == (2, 5)
        assert not torch.isnan(logits).any()
        print(f"  ✓ Sequence length {seq_len}: OK")
    
    print("✓ Variable-length test passed")

def test_preprocessing():
    """Test preprocessing block"""
    print("\nTesting preprocessing block...")
    
    from emmixformer_fixed import PreprocessingBlock
    
    preprocess = PreprocessingBlock(velocity_threshold=40.0)
    
    # Create synthetic eye movement data
    x = torch.randn(4, 512, 2) * 10  # Some random gaze coordinates
    
    slow, fast = preprocess(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Slow stream shape: {slow.shape}")
    print(f"  Fast stream shape: {fast.shape}")
    
    # Check that slow + fast covers all data
    assert slow.shape == fast.shape == (4, 2, 512)
    
    # Check no NaN
    assert not torch.isnan(slow).any()
    assert not torch.isnan(fast).any()
    
    print("✓ Preprocessing block works correctly")

def test_with_nan_input():
    """Test model handles NaN input gracefully"""
    print("\nTesting NaN input handling...")
    
    model = EmMixformer(num_classes=10)
    
    # Create input with some NaN values
    x = torch.randn(4, 512, 2)
    x[0, 100:150, :] = float('nan')  # Add NaN region
    
    # Forward pass
    logits = model(x)
    
    # Output should not contain NaN
    assert not torch.isnan(logits).any(), "Model failed to handle NaN input!"
    
    print("✓ Model handles NaN input correctly")

def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    
    return total, trainable

def main():
    print("="*60)
    print("EmMixformer Model Tests")
    print("="*60)
    
    try:
        # Test 1: Model creation
        model = test_model_creation()
        
        # Test 2: Forward pass
        test_forward_pass(model)
        
        # Test 3: Backward pass
        test_backward_pass(model)
        
        # Test 4: Variable length
        test_variable_length()
        
        # Test 5: Preprocessing
        test_preprocessing()
        
        # Test 6: NaN handling
        test_with_nan_input()
        
        # Count parameters
        count_parameters(model)
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

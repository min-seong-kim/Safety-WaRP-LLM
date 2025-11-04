"""
✅ 올바른 방식: torch.autograd.Function으로 gradient flow 구현
"""
import torch
import torch.nn as nn
from torch import optim

class ReconstructedLinear(torch.autograd.Function):
    """
    Forward: y = (basis_coeff @ U^T) @ x^T + bias
    Backward: gradient flows to basis_coeff
    """
    
    @staticmethod
    def forward(ctx, x, basis_coeff, U, bias):
        """
        Args:
            x: input tensor (batch_size, in_features)
            basis_coeff: learnable parameters (out_features, rank)
            U: fixed basis matrix (in_features, rank)
            bias: bias (out_features)
        
        Returns:
            output: (batch_size, out_features)
        """
        # Reconstruct weight: W = basis_coeff @ U^T
        weight = basis_coeff @ U.T  # (out_features, in_features)
        
        # Forward pass
        output = torch.nn.functional.linear(x, weight, bias)
        
        # Save for backward
        ctx.save_for_backward(x, basis_coeff, U, bias, weight)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: gradient w.r.t. output (batch_size, out_features)
        """
        x, basis_coeff, U, bias, weight = ctx.saved_tensors
        
        # Gradient w.r.t. weight (from linear backward)
        grad_weight = grad_output.T @ x  # (out_features, in_features)
        
        # Gradient w.r.t. basis_coeff: dL/d(basis_coeff) = dL/dW @ (dW/d(basis_coeff))
        # dW/d(basis_coeff) gives us: U^T (because W = basis_coeff @ U^T)
        grad_basis_coeff = grad_weight @ U  # (out_features, rank)
        
        # Gradient w.r.t. U
        grad_U = basis_coeff.T @ grad_weight  # (rank, in_features)
        
        # Gradient w.r.t. input
        grad_x = grad_output @ weight  # (batch_size, in_features)
        
        # Gradient w.r.t. bias
        grad_bias = grad_output.sum(dim=0)
        
        return grad_x, grad_basis_coeff, grad_U, grad_bias


# Test
print("="*60)
print("[TEST 3] Using torch.autograd.Function")
print("="*60)

batch_size = 2
in_features = 10
out_features = 5
rank = 3

# Create tensors
x = torch.randn(batch_size, in_features)
U = torch.randn(in_features, rank)  # Fixed
basis_coeff = nn.Parameter(torch.randn(out_features, rank))  # Learnable
bias = nn.Parameter(torch.randn(out_features))  # Learnable

optimizer = optim.Adam([basis_coeff, bias], lr=0.01)

print(f"[INFO] Shapes:")
print(f"  - x: {x.shape}")
print(f"  - U (fixed): {U.shape}")
print(f"  - basis_coeff (learnable): {basis_coeff.shape}")
print(f"  - bias (learnable): {bias.shape}")

# Forward pass using custom autograd function
y = ReconstructedLinear.apply(x, basis_coeff, U, bias)
print(f"\n[FORWARD]")
print(f"  - output shape: {y.shape}")

loss = y.sum()
print(f"  - loss: {loss.item():.6f}")

# Backward
optimizer.zero_grad()
loss.backward()

print(f"\n[BACKWARD]")
print(f"  - basis_coeff.grad is None: {basis_coeff.grad is None}")
if basis_coeff.grad is not None:
    print(f"  - basis_coeff.grad shape: {basis_coeff.grad.shape}")
    print(f"  - basis_coeff.grad norm: {basis_coeff.grad.norm():.6f}")
    print(f"✅ SUCCESS: Gradient flows to basis_coeff with autograd.Function!")
else:
    print(f"❌ FAILURE")

print(f"  - bias.grad is None: {bias.grad is None}")
if bias.grad is not None:
    print(f"  - bias.grad norm: {bias.grad.norm():.6f}")

# Update
optimizer.step()
print(f"\n[UPDATE]")
print(f"  - parameters updated")

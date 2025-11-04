"""
BasisLinear autograd.Function 테스트
"""
import torch
import torch.nn as nn
from torch import optim

# BasisLinear 복사
class BasisLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, basis_coeff, U, bias, mask):
        """Forward pass"""
        weight = basis_coeff @ U.T
        output = torch.nn.functional.linear(x, weight, bias)
        ctx.save_for_backward(x, basis_coeff, U, bias, weight, mask)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass"""
        x, basis_coeff, U, bias, weight, mask = ctx.saved_tensors
        
        print(f"[Backward Debug]")
        print(f"  - grad_output shape: {grad_output.shape}")
        print(f"  - x shape: {x.shape}")
        print(f"  - basis_coeff shape: {basis_coeff.shape}")
        print(f"  - U shape: {U.shape}")
        print(f"  - weight shape: {weight.shape}")
        
        # Gradient computation
        grad_weight = grad_output.T @ x
        print(f"  - grad_weight shape: {grad_weight.shape}")
        
        grad_basis_coeff = grad_weight @ U
        print(f"  - grad_basis_coeff shape: {grad_basis_coeff.shape}")
        
        # Masking
        mask_broadcast = mask.view(-1, 1)
        grad_basis_coeff = grad_basis_coeff * (1 - mask_broadcast)
        
        grad_U = basis_coeff.T @ grad_weight
        grad_x = grad_output @ weight
        grad_bias = grad_output.sum(dim=0)
        
        print(f"  - grad_x shape: {grad_x.shape}")
        print(f"  - grad_U shape: {grad_U.shape}")
        print(f"  - grad_bias shape: {grad_bias.shape}")
        
        return grad_x, grad_basis_coeff, grad_U, grad_bias, None

# Test
batch_size = 2
in_features = 100
out_features = 50
rank = 10

x = torch.randn(batch_size, in_features)
U = torch.randn(in_features, rank)
basis_coeff = nn.Parameter(torch.randn(out_features, rank))
bias = nn.Parameter(torch.randn(out_features))
mask = torch.randint(0, 2, (out_features,)).float()

print("[Forward]")
y = BasisLinear.apply(x, basis_coeff, U, bias, mask)
print(f"  - output shape: {y.shape}\n")

loss = y.sum()
print("[Backward]")
loss.backward()

print(f"\n[Result]")
print(f"  - basis_coeff.grad is None: {basis_coeff.grad is None}")
if basis_coeff.grad is not None:
    print(f"  - basis_coeff.grad shape: {basis_coeff.grad.shape}")
    print(f"  - basis_coeff.grad norm: {basis_coeff.grad.norm():.6f}")
    print(f"✅ SUCCESS!")

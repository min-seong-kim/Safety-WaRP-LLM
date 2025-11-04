"""
✅ Gradient flow 테스트: basis_coeff.grad가 계산되는지 확인
"""
import torch
import torch.nn as nn
from torch import optim

# 간단한 Linear layer 모델
model = nn.Linear(10, 5)
original_weight = model.weight.data.clone()  # (out_features=5, in_features=10)

# basis_coeff 초기화 (Random U matrix)
# W = basis_coeff @ U^T  =>  (5, 10) = basis_coeff @ U^T
# basis_coeff @ U^T should be (5, 10)
# So basis_coeff is (5, rank), U is (10, rank)
# basis_coeff @ U^T = (5, rank) @ (rank, 10) = (5, 10) ✓
rank = 3
U = torch.randn(10, rank)  # (in_features, rank)
basis_coeff = nn.Parameter(original_weight @ U)  # (5, 10) @ (10, 3) = (5, 3)

print(f"[INFO] Setup:")
print(f"  - original_weight shape: {original_weight.shape}")
print(f"  - U shape: {U.shape}")
print(f"  - basis_coeff shape: {basis_coeff.shape}")

# Optimizer에 추가
optimizer = optim.Adam([basis_coeff], lr=0.01)

# Forward: weight = basis_coeff @ U^T
def forward_with_reconstruction(x):
    """basis_coeff를 사용한 forward pass"""
    reconstructed_weight = basis_coeff @ U.T
    print(f"\n[FORWARD]")
    print(f"  - reconstructed_weight shape: {reconstructed_weight.shape}")
    print(f"  - reconstructed_weight norm: {reconstructed_weight.norm():.6f}")
    
    # Linear 연산
    return torch.nn.functional.linear(x, reconstructed_weight)

# 테스트
x = torch.randn(2, 10)  # batch_size=2, input_dim=10

print(f"\n[TEST 1] Direct forward (basis_coeff @ U^T)")
print(f"  - input shape: {x.shape}")

y = forward_with_reconstruction(x)
print(f"  - output shape: {y.shape}")

loss = y.sum()
print(f"\n[BACKWARD]")
print(f"  - loss: {loss.item():.6f}")

optimizer.zero_grad()
loss.backward()

print(f"  - basis_coeff.grad is None: {basis_coeff.grad is None}")
if basis_coeff.grad is not None:
    print(f"  - basis_coeff.grad shape: {basis_coeff.grad.shape}")
    print(f"  - basis_coeff.grad norm: {basis_coeff.grad.norm():.6f}")
    print(f"✅ SUCCESS: Gradient flows to basis_coeff!")
else:
    print(f"❌ FAILURE: basis_coeff.grad is None!")

# Update
optimizer.step()
print(f"\n[UPDATE]")
print(f"  - basis_coeff updated (step completed)")

# ============================================
# TEST 2: Using hooks (like Phase 3)
# ============================================

print(f"\n" + "="*60)
print(f"[TEST 2] Using forward hook (like Phase 3)")
print(f"="*60)

model2 = nn.Linear(10, 5)
original_weight2 = model2.weight.data.clone()

U2 = torch.randn(10, 3)
basis_coeff2 = nn.Parameter(original_weight2 @ U2)

optimizer2 = optim.Adam([basis_coeff2], lr=0.01)

# Forward hook 등록
def forward_hook(module, input):
    """weight를 basis_coeff @ U^T로 동적 대체"""
    module.weight.data = basis_coeff2 @ U2.T
    print(f"[HOOK] Forward hook executed")
    print(f"  - module.weight shape: {module.weight.shape}")
    return input

model2.register_forward_pre_hook(forward_hook)

print(f"[INFO] Forward hook registered")

x2 = torch.randn(2, 10)
y2 = model2(x2)
print(f"\n[FORWARD] model2(x2)")
print(f"  - output shape: {y2.shape}")

loss2 = y2.sum()

optimizer2.zero_grad()
loss2.backward()

print(f"\n[BACKWARD]")
print(f"  - basis_coeff2.grad is None: {basis_coeff2.grad is None}")
if basis_coeff2.grad is not None:
    print(f"  - basis_coeff2.grad shape: {basis_coeff2.grad.shape}")
    print(f"  - basis_coeff2.grad norm: {basis_coeff2.grad.norm():.6f}")
    print(f"✅ SUCCESS: Gradient flows with hook!")
else:
    print(f"❌ FAILURE: basis_coeff2.grad is None with hook!")

print(f"\n[COMPARISON]")
print(f"  - Test 1 (direct): gradient flows? {basis_coeff.grad is not None}")
print(f"  - Test 2 (hook): gradient flows? {basis_coeff2.grad is not None}")

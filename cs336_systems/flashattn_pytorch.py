import torch

class FlashAttentionCustom(torch.autograd.Function):
    def forward(ctx, Q, K, V, is_causal=False):
        B, N, D = Q.shape
        O = torch.zeros_like(Q)
        L = torch.zeros(B, N)
        ctx.save_for_backward(L)
        return O
    
    def backward(ctx):
        return
import torch
import math

class FlashAttentionCustom(torch.autograd.Function):
    def forward(ctx, Q, K, V, is_causal=False):
        B, N, D = Q.shape
        # attn = torch.nn.functional.softmax(Q @ K.transpose(-2, -1) / math.sqrt(D), dim = -1)
        # O = attn @ V
        B_q = 2
        B_k = 2
        T_q = Q.size(-1) // B_q
        T_k = K.size(-1) // B_k

        O = torch.zeros(B, N, D)
        L = torch.zeros(B, N)

        for i in range(1, T_q):
            Q_i = Q[:,i-1:i * B_q,:]
            m_ij_prev = torch.zeros(B_q)
            l_ij_prev = torch.zeros(B_q)
            O_ij_prev = torch.zeros(B, B_q, D)
            for j in range(1, T_k):
                K_j = K[:, j-1 : j * B_k, :]
                V_j = V[:, j-1 : j * B_k, :]
                S_ij = Q_i @ K_j.transpose(-2, -1) * (D ** 0.5)
                m_ij = max(m_ij_prev, rowmax(S_ij))
                P_ij = exp(S_ij - m_ij) # what is the size
                l_ij = exp(m_ij_prev - m_ij) * l_ij_prev + sum(P_ij, dim=-1)
                O_ij = diag(exp(m_ij_prev - m_ij)) @ O_ij_prev + P_ij @ V_j

                # update
                O_ij_prev = O_ij
                l_ij_prev = l_ij
                m_ij_prev = m_ij
            
                
            L_i = m_ij_prev + log(l_ij_prev)
            O_i = diag(l_ij_prev) ** -1 @ O_ij_prev

            L[:, i-1 : i * B_q, :] = L_i
            O[:,i-1 : i * B_q, :] = O_i

        ctx.save_for_backward(O, L, Q, K, V)
        return O
    
    def backward(ctx):
        return
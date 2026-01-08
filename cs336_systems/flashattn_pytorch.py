import torch
import math

class FlashAttentionCustom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N, D = Q.shape
        # attn = torch.nn.functional.softmax(Q @ K.transpose(-2, -1) / math.sqrt(D), dim = -1)
        # O = attn @ V
        B_q = 16 # tile size for Q (row size)
        B_k = 16 # tile size of K (col size)
        T_q = Q.size(1) // B_q # Total number of rows
        T_k = K.size(1) // B_k 

        O = torch.zeros(B, N, D)
        L = torch.zeros(B, N, 1)

        for i in range(T_q):
            Q_i = Q[:,i*B_q:(i+1)*B_q,:]
            m_ij_prev = torch.zeros(B, B_q, 1)
            l_ij_prev = torch.zeros(B, B_q, 1)
            O_ij_prev = torch.zeros(B, B_q, D)
            for j in range(T_k):
                K_j = K[:, j*B_k:(j+1)*B_k, :]
                V_j = V[:, j*B_k:(j+1)*B_k, :]
                S_ij = Q_i @ K_j.transpose(-2, -1) * (D ** -0.5)
                m_ij = torch.maximum(m_ij_prev, torch.max(S_ij, dim=-1, keepdim=True)[0])
                P_ij = torch.exp(S_ij - m_ij) # S_ij has a row. m_ij is just a value. So pytorch auto broadcasts m_ji to a row vector.
                row_sum = torch.sum(P_ij, dim=-1, keepdim=True)
                l_ij = torch.exp(m_ij_prev - m_ij) * l_ij_prev + torch.sum(P_ij, dim=-1, keepdim=True)
                O_ij = torch.exp(m_ij_prev - m_ij) * O_ij_prev + P_ij @ V_j # the multiplication sign * automatically broadcasts. Same as diag() @ O

                # update
                O_ij_prev = O_ij
                l_ij_prev = l_ij
                m_ij_prev = m_ij
            

            L_i = m_ij_prev + torch.log(l_ij_prev)
            O_i = (l_ij_prev ** -1) * O_ij_prev

            L[:, i*B_q:(i+1)*B_q, :] = L_i
            O[:, i*B_q:(i+1)*B_q, :] = O_i

        # sqeeze L from (B, N, 1) to (B, N)
        L.squeeze_(dim=-1)
        ctx.save_for_backward(O, L, Q, K, V)
        return O
    
    def backward(ctx):
        return


if __name__ == "__main__":
    # use this file: python cs336_systems/flashattn_pytorch.py 
    # run test: pytest -k test_flash_forward_pass_pytorch
    def _make_attn_inputs(device=None):
        torch.random.manual_seed(0)
        batch_size = 4
        n_queries = 4
        n_keys = 4
        D = 8
        q = torch.randn(batch_size, n_queries, D, device=device, requires_grad=True)
        k = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
        v = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
        do = torch.randn(batch_size, n_queries, D, device=device)

        return q, k, v, do
    
    q, k, v, do = _make_attn_inputs()
    O = FlashAttentionCustom.apply(q, k, v)
    print("=== res O", O.shape)
import torch
import torch.nn.functional as F

def flash_attention(Q, K, V, block_size=128):
    B, L, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    O = torch.zeros_like(Q)
    m_i = torch.full((B, L, 1), -float('inf'), device=Q.device)
    l_i = torch.zeros((B, L, 1), device=Q.device)

    for start in range(0, L, block_size):
        end = min(start + block_size, L)
        K_block = K[:, start:end, :]
        V_block = V[:, start:end, :]

        S_ij = torch.bmm(Q, K_block.transpose(1, 2)) * scale
        m_ij = torch.max(S_ij, dim=-1, keepdim=True).values

        m_new = torch.maximum(m_i, m_ij)
        exp1 = torch.exp(m_i - m_new) * l_i
        exp2 = torch.exp(S_ij - m_new)
        l_new = exp1 + exp2.sum(dim=-1, keepdim=True)

        O = (O * torch.exp(m_i - m_new) * (l_i / l_new)) + torch.bmm(exp2 / l_new, V_block)
        m_i, l_i = m_new, l_new

    return O


def standard_attention(Q, K, V):
    D = Q.shape[-1]
    scores = torch.bmm(Q, K.transpose(1, 2)) / (D ** 0.5)
    probs = F.softmax(scores, dim=-1)
    return torch.bmm(probs, V)

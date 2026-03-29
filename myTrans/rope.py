from myTrans.base_params import *

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim=D_K, max_seq_len=BLOCK_SIZE, theta=POS_ENCODING_BASE):
        super().__init__()
        self.dim = dim  # 每个头的维度
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算频率：theta_i = 1/theta^(2i/dim)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 预计算所有位置的cos/sin矩阵（固定值，不训练）
        positions = torch.arange(max_seq_len)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)  # (max_seq_len, dim//2)
        # 扩展到所有维度：偶数维cos，奇数维sin
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def rotate_half(x):
        """核心旋转操作：后一半维度取负，与前一半交叉拼接"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        seq_len = q.shape[-2]
        assert seq_len <= self.max_seq_len, f"seq len {seq_len} more than {self.max_seq_len}"

        # 取对应长度的cos/sin
        cos = self.cos_cached[:seq_len].to(q.device)
        sin = self.sin_cached[:seq_len].to(q.device)

        # 扩展维度适配Q/K：[seq_len, dim] → [1, 1, seq_len, dim]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        # 应用旋转
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin

        return q_rot, k_rot

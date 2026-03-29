from myTrans.rope import *
from myTrans.rms_norm_amp import *

def dot_att(q, k, v, mask=None):
    att_scores = torch.matmul(q, k.transpose(-2, -1))
    att_scores /= np.sqrt(D_K)
    if mask is not None:
        att_scores = att_scores.masked_fill(mask, -1e4)
    att_weight = F.softmax(att_scores, dim=-1)
    att_weight = F.dropout(att_weight, p=DROPOUT_RATE)
    o = torch.matmul(att_weight, v)
    return o, att_weight

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = nn.Linear(D_MODEL, D_MODEL)
        self.linear_k = nn.Linear(D_MODEL, D_MODEL)
        self.linear_v = nn.Linear(D_MODEL, D_MODEL)
        self.linear_o = nn.Linear(D_MODEL, D_MODEL)
        # 新增：千问风格的门控层（轻量化，仅1个线性层）
        # 门控投影到d_k维度（逐头加权），sigmoid激活后和att_o逐元素相乘
        self.gate = nn.Linear(D_K, D_K)  # 维度：d_k → d_k
        # ===== 新增：QK层归一化（核心） =====
        # 对Q/K分别做LayerNorm，维度为d_model（线性投影后的维度）
        # LayerNorm参数：weight (d_model,) + bias (d_model,) → 每层仅2*d_model个参数
        self.norm_q = RMSNorm_AMP(D_K, eps=FP_MIN_EPS_NUM)  # eps避免除0
        self.norm_k = RMSNorm_AMP(D_K, eps=FP_MIN_EPS_NUM)
        # ===== 新增：改用RoPE位置编码 =====
        self.rope = RotaryPositionEmbedding()

    def split_heads(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, NUM_HEADS, D_K)
        x = x.transpose(1, 2)
        return x

    def combine_heads(self, x):
        batch_size = x.shape[0]
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, -1, D_MODEL)
        return x

    def forward(self, q, k, v, mask=None):
        # 1. Linear
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)
        # 2
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        # ===== 新增：QK层归一化 =====
        # 仅对Q/K做归一化，V不做，先拆头在做，各自norm
        q = self.norm_q(q)  # [batch, seq_len, d_model] → 归一化最后一维
        k = self.norm_k(k)
        # 3. attention
        # ===== 新增：应用RoPE =====
        q_rot, k_rot = self.rope(q, k)
        # 替换为框架的实现，可通过flash attention加速（但win上暂时不行）
        att_o = F.scaled_dot_product_attention(
            query=q_rot,
            key=k_rot,
            value=v,
            #attn_mask=mask,  #直接用因果掩码
            dropout_p=DROPOUT_RATE,
            is_causal=True
        )
        # 新增门控层（参考qwen论文）
        gate_weights = self.gate(att_o)  # [batch, n_heads, seq_len, d_k]
        gate_weights = F.gelu(gate_weights)  # 新增轻量级激活
        gate_weights = torch.sigmoid(gate_weights)  # 激活到0~1，动态加权
        # 2. 门控加权：注意力输出 × 门控权重（逐元素相乘）
        att_o = att_o * gate_weights
        # 4
        o = self.combine_heads(att_o)
        # 5. linear
        o = self.linear_o(o)

        return o
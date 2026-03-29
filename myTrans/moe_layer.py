from myTrans.moe_experts import *
from myTrans.multi_att import *
from myTrans.rms_norm_amp import *

class MoEDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.norm1 = RMSNorm_AMP(D_MODEL, eps=FP_MIN_EPS_NUM)
        self.norm2 = RMSNorm_AMP(D_MODEL, eps=FP_MIN_EPS_NUM)
        self.mask_attn = MultiHeadAttention()
        self.gate = nn.Linear(D_MODEL, ROUTE_EXPERTS_NUM)
        self.share_expert_layers = nn.ModuleList([Expert(HIDDEN_SIZE) for _ in range(SHARE_EXPERT_NUM)])
        self.route_expert_layers = nn.ModuleList([Expert(D_MODEL) for _ in range(ROUTE_EXPERTS_NUM)])
        self.padding_id = torch.tensor(PAD_ID, dtype=torch.long)
        self.expert_activations = None  # 存储当前层各路由专家的有效激活次数

    def expert_forward_wrapper(self, expert_layer, expert_x):
        return expert_layer(expert_x)

    def gate_layer(self, inputs):
        batch_size, seq_len, _ = inputs.shape
        # 5.1 计算门控分数（单门控，映射到路由专家数）
        gate_logits = self.gate(inputs)
        gate_scores = torch.softmax(gate_logits, dim=-1)  # 归一化
        # 5.2 处理无效token(Padding)
        token_valid_mask = (inputs != self.padding_id).any(dim=-1)  # [batch, seq_len] → True=有效token
        mask_expanded = token_valid_mask.unsqueeze(-1).expand(gate_scores.shape)
        neg_inf = torch.tensor(-1e4, dtype=gate_scores.dtype, device=gate_scores.device)
        gate_scores = torch.where(mask_expanded, gate_scores, neg_inf)
        # 5.3 Top-K选路由专家
        topk_values, topk_indices = torch.topk(gate_scores, k=ACTIVE_EXPERT_NUM, dim=-1)  # [batch, seq, active_num]
        # 5.4 计算专家输出（共享+路由）
        outputs = torch.zeros_like(inputs)
        # 共享专家：所有有效token都计算，多共享专家则平均
        valid_x = inputs[token_valid_mask]
        share_out = sum(expert(valid_x) for expert in self.share_expert_layers) / len(self.share_expert_layers)
        outputs[token_valid_mask] += share_out
        # 路由专家：按Top-K加权
        flat_x = inputs.reshape(-1, D_MODEL)
        flat_indices = topk_indices.reshape(-1, ACTIVE_EXPERT_NUM)
        flat_scores = gate_scores.reshape(-1, ROUTE_EXPERTS_NUM)

        unique_experts = torch.unique(flat_indices).cpu().tolist()  # 仅获取被选中的专家ID
        logging.debug(f"Active expert num：{len(unique_experts)}/{ROUTE_EXPERTS_NUM}")
        # 新增：初始化专家激活次数（全0）
        self.expert_activations = torch.zeros(ROUTE_EXPERTS_NUM, device=inputs.device)
        flat_valid_mask = token_valid_mask.reshape(-1)
        for idx in unique_experts:
            # 找到选中当前专家的token
            expert_mask = (flat_indices == idx).any(dim=-1)
            if not expert_mask.any():
                continue
            # 新增：仅统计有效token的激活次数
            valid_expert_mask = expert_mask & flat_valid_mask
            self.expert_activations[idx] = valid_expert_mask.sum()  # 记录当前专家的有效激活次数
            # 专家计算+加权
            expert_x = flat_x[expert_mask]
            expert_score = flat_scores[expert_mask, idx:idx + 1]
            # 用Checkpoint包裹专家前向计算
            expert_output = self.route_expert_layers[idx](expert_x)
            """
            expert_output = checkpoint(
                self.expert_forward_wrapper,  # 封装的专家前向函数
                self.route_expert_layers[idx],  # 第idx个专家层（张量外参数，需保证可序列化）
                expert_x,  # 专家输入（张量，Checkpoint会追踪）
                use_reentrant=False,  # PyTorch 2.0+推荐，避免梯度错误
                preserve_rng_state=False  # 不保存随机数状态，提速
            )
            """
            outputs.reshape(-1, D_MODEL)[expert_mask] += expert_output * expert_score

        #return outputs.to(orig_dtype)
        return outputs

    def forward(self, x, mask=None, expert_pool=None):
        # 1. layNorm
        x_norm1 = self.norm1(x)
        # 2. masked att
        o1 = self.mask_attn(q=x_norm1, k=x_norm1, v=x_norm1, mask=mask)
        # 3. 1+x, dropout
        x = x + self.dropout(o1)
        # 4. laynorm
        x_norm2 = self.norm2(x)
        # 5. gate
        g_out = self.gate_layer(x_norm2)
        # 6. 1+x, dropout
        o = x + self.dropout(g_out)
        return o, self.expert_activations
from myTrans.base_params import *

# 第一步：定义带标签平滑的交叉熵损失函数
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, vocab_size, ignore_index=PAD_ID, smoothing=LABEL_SMOOTH):
        super().__init__()
        self.vocab_size = vocab_size  # 词表大小
        self.ignore_index = ignore_index  # 忽略的索引
        self.smoothing = smoothing  # 平滑系数，业界主流0.1（目标token 90%概率）

    def forward(self, pred, target):
        # pred: [B, L, V] 或 [B, V]，target: [B, L] 或 [B]
        # 1. 展平维度，统一处理（适配任意维度）
        pred_flat = pred.reshape(-1, self.vocab_size)  # [N, V]，N=B*L
        target_flat = target.reshape(-1)  # [N]

        # 2. 计算log_softmax
        log_prob = F.log_softmax(pred_flat, dim=-1)  # [N, V]

        # 3. 先过滤ignore_index（关键！避免非法索引）
        mask = (target_flat != self.ignore_index)  # [N]
        pred_flat = pred_flat[mask]  # [M, V]，M为有效token数
        target_flat = target_flat[mask]  # [M]
        log_prob = log_prob[mask]  # [M, V]

        # 4. 构建软标签（dim=-1 对应词表维度）
        soft_target = torch.full_like(log_prob, self.smoothing / (self.vocab_size - 1))
        # 修正：dim=-1，且target_flat.unsqueeze(-1) 形状为 [M, 1]，匹配soft_target
        soft_target.scatter_(-1, target_flat.unsqueeze(-1), 1 - self.smoothing)

        # 5. 计算损失（无除以0风险，因为mask已过滤空值）
        if len(pred_flat) == 0:
            return torch.tensor(0.0, device=pred.device)  # 无有效token时返回0
        loss = -torch.sum(soft_target * log_prob) / len(pred_flat)

        return loss
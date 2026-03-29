from myTrans.ffn import *
from myTrans.multi_att import *
from myTrans.rms_norm_amp import *

class GPTLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_attn = MultiHeadAttention()
        self.ffn = FFN()
        self.norm1 = RMSNorm_AMP(D_MODEL, eps=FP_MIN_EPS_NUM)
        self.norm2 = RMSNorm_AMP(D_MODEL, eps=FP_MIN_EPS_NUM)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x, mask=None):
        # 1
        x_norm1 = self.norm1(x)
        # 2
        o1 = self.mask_attn(x_norm1, x_norm1, x_norm1, mask=mask)
        # 3
        x = x + self.dropout(o1)
        # 4
        x_norm2 = self.norm2(x)
        # 5
        o2 = self.ffn(x_norm2)
        # 用Checkpoint包裹专家前向计算；(按需tradeoff 显存-速度)
        """
        o2 = checkpoint(
            lambda ffn, x: ffn(x),  # 简化的匿名封装函数
            self.ffn,
            x_norm2,  # 输入
            use_reentrant=False  # PyTorch 2.0+推荐，避免梯度错误
        )
        """
        # 6
        o = x + self.dropout(o2)

        return o

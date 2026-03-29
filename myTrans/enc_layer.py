from myTrans.ffn import *
from myTrans.multi_att import *

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention()
        self.ffn = FFN()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x, mask=None):
        # 1
        att_o, att_w = self.self_attn(q=x, k=x, v=x, mask=mask)
        # 2
        o1 = x + self.dropout(att_o)
        # 3
        o1 = self.norm1(o1)
        # 4 ffn
        o2 = self.ffn(o1)
        # 5
        o2 = o1 + self.dropout(o2)
        # 6
        o = self.norm2(o2)
        return o, att_w
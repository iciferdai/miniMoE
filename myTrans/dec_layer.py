from myTrans.ffn import *
from myTrans.multi_att import *

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_att = MultiHeadAttention()
        self.cross_att = MultiHeadAttention()
        self.ffn = FFN()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.norm3 = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x, enc_o, src_mask=None, tgt_mask=None):
        # 1
        att_o1, att_w1 = self.mask_att(q=x, k=x, v=x, mask=tgt_mask)
        # 2
        res_o1 = x + self.dropout(att_o1)
        res_o1 = self.norm1(res_o1)
        # 3
        att_o2, att_w2 = self.cross_att(q=res_o1, k=enc_o, v=enc_o, mask=src_mask)
        # 4
        res_o2 = res_o1 + self.dropout(att_o2)
        res_o2 = self.norm2(res_o2)
        # 5
        o = self.ffn(res_o2)
        # 6
        o = res_o2 + self.dropout(o)
        o = self.norm3(o)
        return o, att_w1, att_w2
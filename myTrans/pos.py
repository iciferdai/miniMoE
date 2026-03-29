from myTrans.base_params import *

class PosEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        pe = torch.zeros(MAX_LEN, D_MODEL)
        pos = torch.arange(0, MAX_LEN, dtype=torch.float).unsqueeze(1)
        # 10000^(2i/d_model)
        div_val = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-math.log(POS_ENCODING_BASE) / D_MODEL))
        # sin & cos
        pe[:,0::2] = torch.sin(pos * div_val)
        pe[:,1::2] = torch.cos(pos * div_val)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # pe shape: [1, max_len, d_model]
        x = x + self.pe[:, :x.shape[1], :]
        return x
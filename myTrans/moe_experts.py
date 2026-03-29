from myTrans.base_params import *

class Expert(nn.Module):
    def __init__(self, d_ff):
        super().__init__()
        self.fc1_linear = nn.Linear(D_MODEL, d_ff, bias=False)
        self.fc2_linear = nn.Linear(d_ff, D_MODEL, bias=False)
        #self.dropout = nn.Dropout(DROPOUT_RATE)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1_linear(x)
        x = self.gelu(x)
        #x = self.dropout(x)   # 小专家不drop了
        x = self.fc2_linear(x)
        return x
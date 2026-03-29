from myTrans.base_params import *

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_linear = nn.Linear(D_MODEL, HIDDEN_SIZE, bias=False)
        self.fc2_linear = nn.Linear(HIDDEN_SIZE, D_MODEL, bias=False)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1_linear(x)
        x = self.gelu(x)
        x = self.dropout(x)
        o = self.fc2_linear(x)
        return o
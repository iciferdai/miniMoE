from myTrans.base_params import *

class RMSNorm_AMP(nn.RMSNorm):
    def forward(self, x):
        # 前向传播时临时将权重转为输入 dtype, 适配AMP混合精度场景，规避warning：Mismatch dtype between input and weight: input dtype = struct c10::Half, weight dtype = float
        weight = self.weight.to(x.dtype) if self.weight is not None else None
        return torch.rms_norm(x, self.normalized_shape, weight, self.eps)
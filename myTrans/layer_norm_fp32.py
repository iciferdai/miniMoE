from myTrans.base_params import *

class LayerNormFP32(nn.LayerNorm):
    def forward(self, x):
        # 将输入转为FP32计算，输出转回原精度（兼容混合精度）
        orig_dtype = x.dtype
        out = super().forward(x.to(torch.float32))
        return out.to(orig_dtype)

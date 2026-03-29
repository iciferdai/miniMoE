from data_dict import *
from myTrans.moe_layer import *
from myTrans.gpt_layer import *
from myTrans.rms_norm_amp import *

class MiniMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_ID)
        #self.pos_embedding = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.gpt_layers = nn.ModuleList([GPTLayer() for _ in range(GPT_LAYER_NUM)])
        self.moe_layers = nn.ModuleList([MoEDecoderLayer() for _ in range(MOE_LAYER_NUM)])

        self.norm = RMSNorm_AMP(D_MODEL, eps=FP_MIN_EPS_NUM)
        # ===== 新增：embedding和LM Head对齐 =====
        #self.scale = 1.0 / math.sqrt(D_MODEL)
        self.lm_head_linear = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, x, mask=None):
        # 1. embedding
        #_, T = x.shape
        #token_embed = self.embedding(x)
        x = self.embedding(x)
        #pos_idx = torch.arange(T, device=token_embed.device)
        #pos_embed = self.pos_embedding(pos_idx)
        #x = token_embed + pos_embed
        for layer in self.gpt_layers:
            x = layer(x, mask=mask)
        all_expert_activations = []
        for layer in self.moe_layers:
            x, layer_activations = layer(x, mask=mask)
            all_expert_activations.append(layer_activations)
        x = self.norm(x)
        # 切换embedding和LM Head对齐，缩小模型后取消对齐，更灵活做后继微调
        #hidden_states_scaled = x * self.scale
        #o = F.linear(hidden_states_scaled, self.embedding.weight, bias=None)
        o = self.lm_head_linear(x)
        return o, all_expert_activations

if __name__ == "__main__":
    print(f"PyTorch 版本：{torch.__version__}")
    print(f"CUDA 版本：{torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"GPU 是否支持 FlashAttention (算力): {torch.cuda.get_device_capability(0)}")
    #print("=== PyTorch 编译配置 ===")
    #for key in torch._C._show_config().split("\n"):
    #    print(key.strip())

    model = MiniMoE()
    model.eval()
    input_ids = torch.randint(0,20,(2,10))
    logits, act = model(input_ids)
    print(f'shape is {logits.shape}')
    print(f'act is {act}')
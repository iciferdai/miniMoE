import os
# 0.92 x 24G = 22.08G
# 0.95 x 24G = 22.8G
#os.environ["PYTORCH_ALLOC_CONF"] = "backend:native,max_split_size_mb:128,per_process_memory_fraction:0.92,garbage_collection_threshold:0.01,roundup_power2_divisions:[256:1,512:2,1024:4,>:8]"
#os.environ["PYTORCH_ALLOC_CONF"] = "backend:cudaMallocAsync,per_process_memory_fraction:0.92"
os.environ["PYTORCH_ALLOC_CONF"] = "backend:cudaMallocAsync"
os.environ["TORCH_COMPILE_CACHE_DIR"] = "E:\\code_space\\miniMoE\\cache_data"

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d|%(levelname)s|%(filename)s:%(lineno)d|%(funcName)s -> %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler
import numpy as np
import math
import gc

DEFAULT_BATCH_SIZE = 256
D_MODEL = 1024
NUM_HEADS = 8
D_K = D_MODEL // NUM_HEADS
assert D_MODEL % NUM_HEADS == 0
HIDDEN_SIZE = D_MODEL * 4
DROPOUT_RATE = 0.1
POS_ENCODING_BASE = 1000.0
# 兼容fp16和fp32
FP_MIN_EPS_NUM = 7e-5

BLOCK_SIZE = 128
IGNORE_INDEX = 3
LABEL_SMOOTH = 0.1

# MoE
SHARE_EXPERT_NUM = 1
ROUTE_EXPERTS_NUM = 47
ACTIVE_EXPERT_NUM = 5
GPT_LAYER_NUM = 2
FROZE_GPT_LAYER_NUM = 2
MOE_LAYER_NUM = 4
FROZE_MOE_LAYER_NUM = 0
GATE_LOSS_WEIGHT = 0.01

CLS_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3
SEP_ID = 4
MASK_ID = 5
UNK_ID = 6
REM_ID_1 = 7
REM_ID_2 = 8
REM_ID_3 = 9
CUS_START_ID = 10
IGN_LOSS_ID = -100

#TEST_PERCENT = 0.2


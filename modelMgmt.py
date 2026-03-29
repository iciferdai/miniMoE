import logging
import signal
from miniMoEModel import *
from processData import *
import torch.optim as optim
from transformers.optimization import Adafactor
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
import math
import time
import matplotlib.pyplot as plt
from SoftLoss import LabelSmoothingCrossEntropy


class ModelManagement():
    def __init__(self, model, train_dataloader, device=torch.device('cpu')):
        # === static ===
        self.STEP_PROGRESS_COUNT = 10
        self.STEP_CHECKPOINT_COUNT = 5000
        self.STEP_IGNORE_CHECKPOINT = 2000
        self.WARMUP_STEPS = 2000
        self.COS_STEPS = 23000
        self.MONITOR_WIN_STEP = 20000
        self.PLT_WIN_REM = 0.1
        # === init ===
        self.model = model
        self.train_dl = train_dataloader
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = GradScaler()
        # === dynamic related with model===
        self.train_loss = float('inf')
        self.best_train_loss = float('inf')
        # === dynamic related with mgmt===
        self.step_count = 0
        self.train_loss_list = []
        self.best_checkpoints = dict()
        self.step_cost = 0.0
        self.data_iter_cost = 0.0
        self.data_trans_cost = 0.0
        self.forward_cost = 0.0
        self.loss_cost = 0.0
        self.backward_cost = 0.0
        self.opt_cost = 0.0
        # === tmp & plt & flags ===
        self.step = 0
        self.steps = 0
        self.monitor_flag = []
        self.fig = None
        self.ax = None
        self.train_line = None
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_f_event = torch.cuda.Event(enable_timing=True)
        self.start_l_event = torch.cuda.Event(enable_timing=True)
        self.start_b_event = torch.cuda.Event(enable_timing=True)
        self.start_o_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        # for manual exit
        self._register_signal_handler()


    def _register_signal_handler(self):
        #  SIGINT(Ctrl+C)   SIGTERM
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle_termination)
        signal.signal(signal.SIGTERM, self._handle_termination)


    def _handle_termination(self, signum, frame):
        print(f"\n!!! CATCH SIGNAL: {signum}, SAVING BEFORE TERMINATING!!!\n...")
        try:
            self.save_checkpoint()
            self.save_state()
            print("SAVED SUCCESS, EXIT NOW")
        except Exception as e:
            print(f"SAVED FAILED: {e}, EXIT")
        finally:
            signal.signal(signal.SIGINT, self.original_sigint)
            signal.signal(signal.SIGTERM, self.original_sigterm)
            exit(0)


    def init_weights(self):
        for name, param in self.model.named_parameters():
            #print(f'INIT {name}: {param.shape}')
            if 'weight' in name and 'emb' in name:
                # 嵌入层初始化
                nn.init.normal_(param, mean=0.0, std=1.0 / math.sqrt(D_MODEL))
            elif 'weight' in name and 'linear' in name:
                # 注意力层Linear
                if 'attn' in name:
                    nn.init.xavier_uniform_(param)
                # FFN层Linear
                elif 'ffn' in name or 'expert' in name:
                    nn.init.kaiming_uniform_(param, nonlinearity='relu',a=math.sqrt(2 / 1.7015))
                else:
                    # 通用Linear层：保留框架默认初始化
                    logging.info(f'unexpected name: {name}, pass')


    def preallocate_gpu_memory(self, target_gb=20.0):
        device = torch.device("cuda")
        target_bytes = int(target_gb * 1024**3)
        dummy = torch.zeros((target_bytes // 2,), dtype=torch.float16, device=device)
        del dummy
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        free_in_pool = reserved - allocated
        logging.info(f"preallocate_gpu_memory reserved: {reserved:.2f}GB, allocated: {allocated:.2f}GB, free_in_pool: {free_in_pool:.2f}GB")


    def init_train(self, is_fp16=False, disable_compile=False, is_tf32=True):
        #self.preallocate_gpu_memory(22.0)
        if is_tf32:
            # 老API(先兼容)
            #torch.backends.cuda.matmul.allow_tf32 = True
            #torch.backends.cudnn.allow_tf32 = True
            # 新API
            torch.backends.fp32_precision = "tf32"
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.fp32_precision = "tf32"
        else:
            #torch.backends.cuda.matmul.allow_tf32 = False
            #torch.backends.cudnn.allow_tf32 = False
            torch.backends.fp32_precision = "ieee"
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
        # 启用FlashAttention（显式设置，但win上暂时不支持）
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logging.info(f'Check Flash Attention On: {torch.backends.cuda.is_flash_attention_available()}')

        # win上没有triton，暂无法使用编译加速
        #if not disable_compile: logging.info('Start compile model...')
        #self.model = torch.compile(self.model, disable=disable_compile)
        self.model.to(self.device)
        self.model.train()
        self.init_dashboard()
        # 分组设置weight_decay
        param_groups = [
            # 权重参数：应用weight_decay
            {'params': [p for n, p in self.model.named_parameters() if 'weight' in n and 'norm' not in n],
             'weight_decay': 0.03},
            # 偏置/归一化参数：无weight_decay
            {'params': [p for n, p in self.model.named_parameters() if 'bias' in n or 'norm' in n],
             'weight_decay': 0.0}
        ]
        self.optimizer = optim.AdamW(param_groups,lr=5e-5, betas=(0.9, 0.95), eps=1e-8, fused=True)
        #self.optimizer = Adafactor(param_groups, lr=None, eps=(FP_MIN_EPS_NUM, 1e-3), clip_threshold=1.0,
        #                           beta1=None, weight_decay=None, scale_parameter=True, relative_step=True, warmup_init=True)
        #self.optimizer = Adafactor(param_groups, lr=3e-4, eps=(FP_MIN_EPS_NUM, 1e-3), clip_threshold=1.0,
        #                            beta1=0.9, weight_decay=None, scale_parameter=True, relative_step=False, warmup_init=False)
        #if not disable_compile: logging.info('Start compile optimizer...')
        #self.optimizer.step = torch.compile(self.optimizer.step,backend="cudagraphs", disable=disable_compile)
        self.criterion = LabelSmoothingCrossEntropy(VOCAB_SIZE)
        #self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)


    def init_sft_train(self, weight_path, is_tf32=True):
        if is_tf32:
            # 老API(先兼容)
            #torch.backends.cuda.matmul.allow_tf32 = True
            #torch.backends.cudnn.allow_tf32 = True
            # 新API
            torch.backends.fp32_precision = "tf32"
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.fp32_precision = "tf32"
        else:
            #torch.backends.cuda.matmul.allow_tf32 = False
            #torch.backends.cudnn.allow_tf32 = False
            torch.backends.fp32_precision = "ieee"
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
        # 启用FlashAttention（显式设置，但win上暂时不支持）
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logging.info(f'Check Flash Attention On: {torch.backends.cuda.is_flash_attention_available()}')

        for num, layer in enumerate(self.model.gpt_layers):
            if num < FROZE_GPT_LAYER_NUM:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                break

        for num, layer in enumerate(self.model.moe_layers):
            if num < FROZE_MOE_LAYER_NUM:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                break

        for param in self.model.embedding.parameters():
            param.requires_grad = False

        self.model.to(self.device)
        self.model.train()
        self.load_checkpoint(weight_path, only_weights=False)
        self.init_dashboard()

        param_groups = [
            # 权重参数：应用weight_decay
            {
                'params': [p for n, p in self.model.named_parameters() if
                           'weight' in n and 'norm' not in n and p.requires_grad],
                'weight_decay': 0.01
            },
            # 偏置/归一化参数：无weight_decay
            {
                'params': [p for n, p in self.model.named_parameters() if
                           ('bias' in n or 'norm' in n) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        #for n, p in self.model.named_parameters():
        #    print(f'Parameter Name: {n}; gard: {p.requires_grad}')

        self.optimizer = optim.AdamW(param_groups, lr=5e-5, betas=(0.9, 0.95), eps=1e-8, fused=True)
        #self.optimizer = Adafactor(param_groups, lr=None, eps=(FP_MIN_EPS_NUM, 1e-3), clip_threshold=1.0,
        #                           beta1=None, scale_parameter=True, relative_step=True, warmup_init=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=IGN_LOSS_ID)
        #self.criterion = LabelSmoothingCrossEntropy(VOCAB_SIZE, ignore_index=IGN_LOSS_ID)


    def init_eval(self):
        self.model.eval()
        self.model.to(self.device)


    def progress_info(self, force=False):
        log_flag = False
        monitor_info=''
        if self.monitor_flag:
            monitor_info=f"Monitor: {','.join(self.monitor_flag)}"
            self.monitor_flag = []
            log_flag = True

        cost_info=''
        if self.step % self.STEP_PROGRESS_COUNT == 0:
            step_cpu_avg = self.step_cost / self.STEP_PROGRESS_COUNT
            step_data_avg = self.data_iter_cost / self.STEP_PROGRESS_COUNT
            gpu_trans_avg = self.data_trans_cost / self.STEP_PROGRESS_COUNT
            gpu_forward_avg = self.forward_cost / self.STEP_PROGRESS_COUNT
            gpu_loss_avg = self.loss_cost / self.STEP_PROGRESS_COUNT
            gpu_backward_avg = self.backward_cost / self.STEP_PROGRESS_COUNT
            gpu_opt_avg = self.opt_cost / self.STEP_PROGRESS_COUNT
            gpu_cpu_pct = (gpu_trans_avg + gpu_forward_avg + gpu_loss_avg + gpu_backward_avg + gpu_opt_avg) * 100 / step_cpu_avg
            self.step_cost = 0.0
            self.data_iter_cost = 0.0
            self.data_trans_cost = 0.0
            self.forward_cost = 0.0
            self.loss_cost = 0.0
            self.backward_cost = 0.0
            self.opt_cost = 0.0
            self.update_dashboard()
            cost_info = f'Step Cost(AVG ms): {step_cpu_avg:.3f}, GPU/Step Percent: {gpu_cpu_pct:.2f}%\n'
            cost_info += f'Each Cost(AVG ms): {step_data_avg:.3f}|{gpu_trans_avg:.3f}|{gpu_forward_avg:.3f}|{gpu_loss_avg:.3f}|{gpu_backward_avg:.3f}|{gpu_opt_avg:.3f}\n'
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
            #max_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3   # 这个意义不大
            free_in_pool = reserved - allocated
            cost_info += f'GPU Mem Mgmt: Allocated {allocated:.2f}GB(Max: {max_allocated:.2f}GB), Reserved {reserved:.2f}GB, Pool Left: {free_in_pool:.2f}GB'
            log_flag = True

        force_info=''
        if force:
            force_info = f'Best checkpoints: {self.best_checkpoints}'
            log_flag = True

        if log_flag:
            log_info=f'[{self.step}/{self.steps}]|Step_{self.step_count}] Progress Info:\n'
            lr=self.optimizer.param_groups[0]['lr']
            if not lr: lr=0.0
            loss_info=f'Loss: {self.train_loss:.4f}, Best loss: {self.best_train_loss:.4f}, lr: {lr:.6f}'
            log_info+=loss_info
            if monitor_info: log_info = log_info + '\n' + monitor_info
            if cost_info: log_info = log_info + '\n' + cost_info
            if force_info: log_info = log_info + '\n' + force_info
            logging.info(log_info)
            # 不每轮更新图表，太占资源，复用log打印节奏更新
            self.update_dashboard(True)
        else:
            # 正常的update只处理窗口互动事件，避免窗口卡死
            self.update_dashboard(False)

        if not force and (self.step_count % self.STEP_CHECKPOINT_COUNT == 0):
            self.save_checkpoint()
            self.save_state()


    def save_checkpoint(self, ckp_name='', enable_log=True):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'best_train_loss': self.best_train_loss
        }
        if ckp_name:
            weight_path = './saves/' + ckp_name
        else:
            weight_path = f'./saves/CheckPoint_Ep{self.step_count}_{self.train_loss:.4f}.pth'
        torch.save(checkpoint, weight_path)
        if enable_log: logging.info(f"checkpoint: {weight_path} Saved")


    def load_checkpoint(self, ckp_name='', only_weights=False):
        if not ckp_name:
            print('No checkpoint provided.')
            return
        weight_path = './saves/' + ckp_name
        try:
            ckpt = torch.load(weight_path, map_location=self.device)
            # 废弃的权重
            #unused_keys = ["pos_embedding.weight"]
            #filtered_state_dict = {k: v for k, v in ckpt["state_dict"].items() if k not in unused_keys}
            #self.model.load_state_dict(filtered_state_dict)
            self.model.load_state_dict(ckpt["state_dict"])
            if not only_weights:
                #self.optimizer.load_state_dict(ckpt["optimizer"])
                self.train_loss = ckpt["train_loss"]
                self.best_train_loss = ckpt["best_train_loss"]
            logging.info(f"checkpoint: {weight_path} Loaded")
        except Exception as e:
            logging.error(f"load_checkpoint Error: {e}", exc_info=True)


    def save_state(self, state_name='', enable_log=True):
        manager_state = {
            'step_count': self.step_count,
            'train_loss_list': self.train_loss_list,
            'best_checkpoints': self.best_checkpoints
        }
        if state_name:
            state_path = './saves/' + state_name
        else:
            state_path = f'./saves/State_Ep{self.step_count}_{self.best_train_loss:.4f}.pkl'
        with open(state_path, "wb") as f:
            pickle.dump(manager_state, f)
        if enable_log: logging.info(f"State saved at {state_path}")


    def load_state(self, state_name=''):
        if not state_name:
            print('No state provided.')
            return
        state_path = './saves/' + state_name
        try:
            with open(state_path, 'rb') as f:
                manager_state = pickle.load(f)
                self.step_count = manager_state['step_count']
                self.train_loss_list = manager_state['train_loss_list']
                self.best_checkpoints = manager_state['best_checkpoints']
                logging.info(f"State: {state_path} Loaded")
        except Exception as e:
            logging.error(f"load_state Error: {e}", exc_info=True)


    def clear_state(self):
        self.train_loss = float('inf')
        self.best_train_loss = float('inf')
        self.step_count = 0
        self.train_loss_list = []
        self.best_checkpoints = dict()


    def save_best(self):
        self.save_checkpoint('best_loss_cpt.pth', False)
        self.best_checkpoints[self.step_count] = self.train_loss
        self.save_state('best_state.pkl', False)


    def load_best(self):
        self.load_checkpoint('best_loss_cpt.pth', False)
        self.load_state('best_state.pkl')
        logging.info(f'Load Best; Best checkpoints: {self.best_checkpoints}')


    def roll_back(self, with_state=False):
        self.load_checkpoint('best_loss_cpt.pth', False)
        if with_state: self.load_state('best_state.pkl')


    def trans_data2dev(self, *args):
        transferred_args = []
        for arg in args:
            try:
                transferred_arg = arg.to(self.device, non_blocking=True)
                transferred_args.append(transferred_arg)
            except Exception as e:
                logging.error(f"trans_data2dev Error: {e}", exc_info=True)
                transferred_args.append(arg)

        if len(transferred_args) == 1:
            return transferred_args[0]
        return tuple(transferred_args)


    def init_dashboard(self):
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # ion
        plt.switch_backend('TkAgg')
        plt.ion()
        # draw
        # 更新，改用滚动窗口监测后，不需要叠加看log窗口了
        self.fig, self.ax = plt.subplots(figsize=(20, 6), num="Loss Dashboard")
        self.ax.set_xlabel('Steps')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Linear Scale Loss')
        self.ax.grid(alpha=0.2)
        # init
        self.train_line, = self.ax.plot(range(1, self.step_count+1),
                                        self.train_loss_list,
                                        label='Train Loss',
                                        marker='o',
                                        markersize=2)
        # update
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def update_dashboard(self, update_data=False):
        if update_data:
            loss_list_len = len(self.train_loss_list)
            if loss_list_len <= self.MONITOR_WIN_STEP:
                x_data = range(1, loss_list_len + 1)
                y_data = self.train_loss_list
                self.ax.set_xlim(left=0, right=loss_list_len + 2)
            else:
                x_start = loss_list_len - self.MONITOR_WIN_STEP + 1
                x_end = loss_list_len + 1
                x_data = range(x_start, x_end)
                y_data = self.train_loss_list[-self.MONITOR_WIN_STEP:]
                self.ax.set_xlim(left=x_start-1, right=x_end + 1)
            # set data
            self.train_line.set_xdata(x_data)
            self.train_line.set_ydata(y_data)
            # 自己计算效率高，上下预留，作为loss先不用考虑负数
            y_min = min(y_data) * (1-self.PLT_WIN_REM)
            y_max = max(y_data) * (1+self.PLT_WIN_REM)
            self.ax.set_ylim(bottom=y_min, top=y_max)
            # update
            #self.ax.legend()
            #self.fig.canvas.draw()
            self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        # optional
        #time.sleep(0.001)


    def show_dashboard(self):
        plt.ioff()
        plt.show()


    def loss_algorithm(self):
        if self.train_loss < self.best_train_loss and self.step_count > self.STEP_IGNORE_CHECKPOINT:
            self.save_best()
            self.monitor_flag.append(f'Save Best Loss! ({self.best_train_loss:.4f}->{self.train_loss:.4f})')
            self.best_train_loss = self.train_loss


    def loss_fn(self, output, batch_y):
        # 输出截取：[BS, BLOCK, VOCAB_SIZE] → [BS, BLOCK-IGNORE_INDEX, VOCAB_SIZE]
        out_ignore = output[:, IGNORE_INDEX:, :]
        # 目标值截取：[BS, BLOCK] → [BS, BLOCK-IGNORE_INDEX]
        tgt_ignore = batch_y[:, IGNORE_INDEX:]
        # 展平计算损失
        out_flat = out_ignore.reshape(-1, VOCAB_SIZE)
        tgt_flat = tgt_ignore.reshape(-1)
        loss = self.criterion(out_flat, tgt_flat)
        return loss


    def cal_gate_loss(self, all_expert_activations):
        # 计算均衡损失
        balance_loss = torch.tensor(0.0, device=self.device)
        if self.model.training and len(all_expert_activations) > 0:
            # 汇总所有MoE层的激活次数
            total_activations = sum(all_expert_activations)
            total_tokens = total_activations.sum()
            if total_tokens > 0 and not torch.isinf(total_tokens):
                freq = total_activations / total_tokens  # 激活频率归一化
                balance_loss = torch.var(freq) * GATE_LOSS_WEIGHT  # 方差损失，权重0.01
        return balance_loss


    def get_batch_loss(self, one_pack_data, is_fp16=False):
        # unpack: Source -> processData
        batch_x, batch_y = one_pack_data
        self.start_event.record()
        batch_x, batch_y = self.trans_data2dev(batch_x, batch_y)
        # forward
        if is_fp16:
            with autocast(device_type='cuda', dtype=torch.float16):
                self.start_f_event.record()
                output, all_expert_activations = self.model(batch_x)
                self.start_l_event.record()
                loss = self.loss_fn(output, batch_y)
                balance_loss = self.cal_gate_loss(all_expert_activations)
                loss += balance_loss
        else:
            self.start_f_event.record()
            output, all_expert_activations = self.model(batch_x)
            self.start_l_event.record()
            loss = self.loss_fn(output, batch_y)
        return loss


    def train_epoch(self, is_fp16=False):
        loader_iter = iter(self.train_dl)
        while True:
            # ===================== Step开始 =====================
            # CPU墙钟时间：记录整个step的开始（包含数据加载、CPU预处理等）
            start_time = time.time()
            self.optimizer.zero_grad()

            # ===================== 数据加载（CPU） =====================
            one_pack_data = next(loader_iter)
            get_batch_time = time.time()

            # ===================== 含传数据、计算loss =====================
            loss = self.get_batch_loss(one_pack_data, is_fp16)

            # ===================== 反向+优化器 =====================
            if is_fp16:
                self.start_b_event.record()
                self.scaler.scale(loss).backward()  # 缩放loss并反向传播
                self.start_o_event.record()
                self.scaler.unscale_(self.optimizer)  # 反缩放梯度（用于梯度裁剪）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
                self.scaler.step(self.optimizer)  # 优化器更新（自动处理梯度缩放）
                self.scaler.update()  # 更新scaler的缩放因子
                self.end_event.record()
            else:
                self.start_b_event.record()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 不手动梯度裁剪
                self.start_o_event.record()
                self.optimizer.step()
                self.end_event.record()

            # ===================== 学习率调度器 =====================
            if self.scheduler is not None: self.scheduler.step()

            # ===================== 同步，清理，计时，计数 =====================
            torch.cuda.synchronize()
            self.train_loss = loss.item()
            # 主动清理GPU变量，释放显存
            del loss
            gc.collect()

            gpu_trans_ms = self.start_event.elapsed_time(self.start_f_event)
            gpu_forward_ms = self.start_f_event.elapsed_time(self.start_l_event)
            gpu_loss_ms = self.start_l_event.elapsed_time(self.start_b_event)
            gpu_backward_ms = self.start_b_event.elapsed_time(self.start_o_event)
            gpu_opt_ms = self.start_o_event.elapsed_time(self.end_event)
            step_data_ms = (get_batch_time - start_time) * 1000
            step_cpu_ms = (time.time() - start_time) * 1000
            #print(f'Step cost(ms): {step_cpu_ms:.3f}, Each: {step_data_ms:.3f}|{gpu_trans_ms:.3f}|{gpu_forward_ms:.3f}|{gpu_loss_ms:.3f}|{gpu_backward_ms:.3f}|{gpu_opt_ms:.3f}')
            self.step_cost += step_cpu_ms
            self.data_iter_cost += step_data_ms
            self.data_trans_cost += gpu_trans_ms
            self.forward_cost += gpu_forward_ms
            self.loss_cost += gpu_loss_ms
            self.backward_cost += gpu_backward_ms
            self.opt_cost += gpu_opt_ms
            # ===================== 管理类记录，保存，log等，不计入耗时 =====================
            self.train_loss_list.append(self.train_loss)
            self.loss_algorithm()
            self.step += 1
            self.step_count += 1
            self.progress_info()
            if self.step >= self.steps: break

    def train_one_sft_step(self, one_pack_data, is_fp16):
        # ===================== 含传数据、计算loss =====================
        sft_loss = self.get_batch_loss(one_pack_data, is_fp16)
        # ===================== 反向+优化器 =====================
        if is_fp16:
            self.start_b_event.record()
            self.scaler.scale(sft_loss).backward()  # 缩放loss并反向传播
            self.start_o_event.record()
            #self.scaler.unscale_(self.optimizer)  # 反缩放梯度（用于梯度裁剪）
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
            self.scaler.step(self.optimizer)  # 优化器更新（自动处理梯度缩放）
            self.scaler.update()  # 更新scaler的缩放因子
            self.end_event.record()
        else:
            self.start_b_event.record()
            sft_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 不手动梯度裁剪
            self.start_o_event.record()
            self.optimizer.step()
            self.end_event.record()

        # ===================== 学习率调度器 =====================
        if self.scheduler is not None: self.scheduler.step()

        # ===================== 同步，清理，计时，计数 =====================
        torch.cuda.synchronize()
        self.train_loss = sft_loss.item()
        # 主动清理GPU变量，释放显存
        del sft_loss
        gc.collect()

        gpu_trans_ms = self.start_event.elapsed_time(self.start_f_event)
        gpu_forward_ms = self.start_f_event.elapsed_time(self.start_l_event)
        gpu_loss_ms = self.start_l_event.elapsed_time(self.start_b_event)
        gpu_backward_ms = self.start_b_event.elapsed_time(self.start_o_event)
        gpu_opt_ms = self.start_o_event.elapsed_time(self.end_event)
        # print(f'Step cost(ms): {step_cpu_ms:.3f}, Each: {step_data_ms:.3f}|{gpu_trans_ms:.3f}|{gpu_forward_ms:.3f}|{gpu_loss_ms:.3f}|{gpu_backward_ms:.3f}|{gpu_opt_ms:.3f}')
        self.data_trans_cost += gpu_trans_ms
        self.forward_cost += gpu_forward_ms
        self.loss_cost += gpu_loss_ms
        self.backward_cost += gpu_backward_ms
        self.opt_cost += gpu_opt_ms


    def train_steps(self, input_steps, is_fp16=False):
        if self.optimizer is None:
            logging.error('NEED INIT TRAIN FIRST!!!')
            return
        self.steps = input_steps

        if input_steps > self.WARMUP_STEPS + self.COS_STEPS:
            warm_steps = self.WARMUP_STEPS
            cos_steps = self.COS_STEPS
            last_step = input_steps - self.WARMUP_STEPS - self.COS_STEPS
        else:
            warm_steps = 1
            cos_steps = 1
            last_step = input_steps-2
        
        print(f'Strat Training with policy: Warm-{warm_steps}，Cos-{cos_steps}, Last-{last_step}')
        warmup = LinearLR(self.optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warm_steps)
        cos = CosineAnnealingLR(self.optimizer,T_max=cos_steps,eta_min=1e-6)
        constant = ConstantLR(self.optimizer,factor=0.02,total_iters=last_step)
        self.scheduler = SequentialLR(self.optimizer,schedulers=[warmup, cos, constant],milestones=[warm_steps, warm_steps+cos_steps])

        # 单epoch无限循环
        self.train_epoch(is_fp16)
        self.save_checkpoint()


    def train_sft_steps(self, input_steps, is_fp16=False, is_scheduler=True):
        if self.optimizer is None:
            logging.error('NEED INIT TRAIN FIRST!!!')
            return
        self.steps = input_steps

        if is_scheduler:
            if input_steps > self.WARMUP_STEPS + self.COS_STEPS:
                warm_steps = self.WARMUP_STEPS
                cos_steps = self.COS_STEPS
                last_step = input_steps - self.WARMUP_STEPS - self.COS_STEPS
            else:
                warm_steps = 1
                cos_steps = 1
                last_step = input_steps - 2

            print(f'Strat SFT Training with policy: Warm-{warm_steps}，Cos-{cos_steps}, Last-{last_step}')
            warmup = LinearLR(self.optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warm_steps)
            cos = CosineAnnealingLR(self.optimizer, T_max=cos_steps, eta_min=1e-6)
            constant = ConstantLR(self.optimizer, factor=0.02, total_iters=last_step)
            self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cos, constant],
                                          milestones=[warm_steps, warm_steps + cos_steps])
        else:
            print(f'Start train_sft_steps without scheduler')

        # 按step训练但计算epoch
        epoch_count = 0
        while True:
            start_time = time.time()
            for one_pack_data in self.train_dl:
                get_batch_time = time.time()
                self.optimizer.zero_grad()
                self.train_one_sft_step(one_pack_data, is_fp16)
                self.loss_algorithm()
                self.step += 1
                self.step_count += 1
                step_data_ms = (get_batch_time - start_time) * 1000
                step_cpu_ms = (time.time() - start_time) * 1000
                self.step_cost += step_cpu_ms
                self.data_iter_cost += step_data_ms
                # ===================== 管理类记录，保存，log等，不计入耗时 =====================
                self.train_loss_list.append(self.train_loss)
                self.progress_info()

                if self.step >= self.steps: break
                start_time = time.time()
            epoch_count += 1
            logging.info(f'Finished epoch {epoch_count}, step {self.step}')
            if self.step >= self.steps: break

        self.save_checkpoint()


    def predict_step(self, id_dict):
        x_tensor = self.trans_data2dev(torch.tensor([id_dict], dtype=torch.long))
        output, _ = self.model(x_tensor)
        #[BS, BLOCK, VOCAB_SIZE]，预测最后1位
        last_token_logits = output[:, -1, :]
        return last_token_logits


    def predict_best(self, id_dict):
        pred_text = []
        prob_list = []
        prob_log_list = []
        for i in range(BLOCK_SIZE - len(id_dict) - 1):
            last_token_logits = self.predict_step(id_dict)
            # BEST
            next_token_idx = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            probs = F.softmax(last_token_logits, dim=-1)
            max_prob = torch.gather(probs, dim=-1, index=next_token_idx)
            token_log_prob = torch.log(max_prob + FP_MIN_EPS_NUM).item()
            # next id
            next_id = next_token_idx[0].item()
            id_dict.append(next_id)
            pred_text.append(idx2token[next_id])
            prob_list.append(max_prob[0].item())
            prob_log_list.append(token_log_prob)
            # EOS
            if next_id == EOS_ID:
                break

        output_text = ''.join(pred_text)
        return output_text, prob_list, prob_log_list

    def predict_top_k(self, id_dict, temperature, k):
        pred_text = []
        prob_list = []
        prob_log_list = []
        for i in range(BLOCK_SIZE - len(id_dict) - 1):
            last_token_logits = self.predict_step(id_dict)
            # temperature
            logits = last_token_logits / temperature
            probs = F.softmax(logits, dim=-1)
            # top-k
            topk_probs, topk_ids = torch.topk(probs, k=k, dim=-1)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
            sample_idx = torch.multinomial(topk_probs, num_samples=1)
            # next id
            sample_idx_scalar = sample_idx[0].item()
            next_id_tensor = topk_ids[0][sample_idx_scalar].unsqueeze(0).unsqueeze(0)
            next_id = next_id_tensor.item()
            sample_prob = torch.gather(probs, dim=-1, index=next_id_tensor)
            token_log_prob = torch.log(sample_prob + FP_MIN_EPS_NUM).item()
            id_dict.append(next_id)
            pred_text.append(idx2token[next_id])
            prob_list.append(sample_prob[0].item())
            prob_log_list.append(token_log_prob)

            if next_id == EOS_ID:
                break

        output_text = ''.join(pred_text)
        return output_text, prob_list, prob_log_list


    def predict_manual(self, txt, mode='TOP_K', temperature = 1.0, top_k = 3, is_prob=False):
        # pre-process
        id_dict = []
        cn_list = list(txt)
        for i in cn_list:
            if i in token2idx.keys():
                id_dict.append(token2idx[i])
            else:
                id_dict.append(UNK_ID)
        id_dict.append(SEP_ID)

        with torch.no_grad():
            # 省略，仅做2个模式，且优先TopK
            if mode == 'TOP_K':
                print(f'Use mode TOP-k({temperature}|{top_k}) Predict:')
                output_text, prob_list, prob_log_list = self.predict_top_k(id_dict, temperature, top_k)
            else:
                logging.warning(f'Not TOP-k mode: {mode}, use BEST!')
                output_text, prob_list, prob_log_list = self.predict_best(id_dict)

            print(f'{txt} | {output_text}')
            if is_prob:
                print(','.join([f"{num:.2f}" for num in prob_list]))
                print(f"Sum prob: {(sum(prob_log_list) / len(prob_log_list)):.2f}")


if __name__ == '__main__':
    print('init model...')
    model = MiniMoE()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('init ModelManagement...')
    m_mgmt = ModelManagement(model, None, dev)
    print('Empty, Do nothing, Exit...')
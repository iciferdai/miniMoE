from data_dict import *
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, ConcatDataset
import re
import random
import json
import pickle
import array

class PklDataset(Dataset):
    def __init__(self, file_path_list, change_file_count=2000):
        super().__init__()
        self.file_path_list = file_path_list
        self.total_files_num = len(file_path_list)
        self.current_array = None
        self.current_array_len = 0
        self.max_valid_idx = 0
        self.pop_count = 0
        self.change_file_count = change_file_count
        self.limit_len = 100000000
        self.block_size = BLOCK_SIZE

    def __len__(self):
        return self.limit_len

    def _load_pkl_file(self, f_p):
        logging.debug(f"PklDataset: {f_p} loading")
        try:
            with open(f_p, "rb") as f:
                d_l = pickle.load(f)
            logging.debug(f"PklDataset: {f_p} loaded")
        except Exception as e:
            logging.error(f"PklDataset: Load {f_p} Failed: {e}")
            self.current_array = None
            self.current_array_len = 0
            d_l = []
        return d_l

    def __getitem__(self, idx):
        if self.pop_count % self.change_file_count == 0:
            self.current_array = None
            gc.collect()

            file_idx = random.randint(0, self.total_files_num - 1)
            self.current_array = self._load_pkl_file(self.file_path_list[file_idx])
            self.current_array_len = len(self.current_array)
            self.max_valid_idx = self.current_array_len - self.block_size - 1

        random_idx = random.randint(0, self.max_valid_idx)
        self.pop_count += 1

        src = self.current_array[random_idx : random_idx + self.block_size]
        tgt = self.current_array[random_idx + 1 : random_idx + self.block_size + 1]

        src_tensor = torch.tensor(src, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt, dtype=torch.long)

        return src_tensor, tgt_tensor

def process_data():
    file_path_list = []
    for i in range(22):
        array_file = f'./cache_data/news_array_int16_chunk_{i}.pkl'
        file_path_list.append(array_file)

    dataloader = DataLoader(
        dataset=PklDataset(file_path_list),
        batch_size=DEFAULT_BATCH_SIZE,
        persistent_workers=True,
        shuffle=False,  # dataset内部已随机化
        drop_last=True,
        num_workers=6,  # 多线程读取，把IO耗时分摊掉
        pin_memory=True,
        prefetch_factor=2  # 预加载2批数据，进一步减少等待
    )
    return dataloader


def load_sft_file(f_p):
    try:
        with open(f_p, "rb") as f:
            d_l = pickle.load(f)
        logging.info(f"Pkl Dataset: {f_p} loaded")
    except Exception as e:
        logging.error(f"Pkl Dataset: {f_p} Failed: {e}")
        return None
    return d_l

def process_sft_data():
    #src_file = f'./cache_data/sft_qa_ids_src.pkl'
    #tgt_file = f'./cache_data/sft_qa_ids_tgt.pkl'
    src_file = f'./cache_data/sft_pc_qa_ids_src.pkl'
    tgt_file = f'./cache_data/sft_pc_qa_ids_tgt.pkl'
    src_list = load_sft_file(src_file)
    tgt_list = load_sft_file(tgt_file)
    src_tensor = torch.tensor(src_list, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_list, dtype=torch.long)
    dataset = TensorDataset(src_tensor, tgt_tensor)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2  # 预加载2批数据，进一步减少等待
    )
    return dataloader



"""
def load_line_no(file_path):
    try:
        line_pos = []
        current_pos = 0
        with open(file_path, "rb") as file:
            while True:
                line = file.readline()  # 按行读取（仅移动指针，不加载内容到内存）
                if not line:
                    break
                line_pos.append(current_pos)  # 记录该行的起始字节位置
                current_pos += len(line)  # 更新下一行的起始位置
    except Exception as e:
        print(f"json parse Error：{e}")
    return line_pos

class LineJsonDataset(Dataset):
    def __init__(self, file_path, line_pos):
        self.file_path = file_path
        self.line_pos = line_pos
        self.mask = generate_default_mask(BLOCK_SIZE, NUM_HEADS)
        self.total_lines = len(self.line_pos)
        # 文本过滤, 预编译正则
"""
        #allowed_pattern = r'[^\u4e00-\u9fa5A-Za-z0-9。！？；，：、…—”“（）《》·「」『』【】·,.!?;:(){}[\]<>"\'’‘-]'
"""
        self.pattern = re.compile(allowed_pattern)
        # 预构建按长度分组的token集合（仅初始化时执行一次）
        self.token2idx = token2idx
        self.token_sets = {1: set(), 2: set(), 3: set(), 4: set()}
        for token, idx in token2idx.items():
            token_len = len(token)
            if 1 <= token_len <= 4: self.token_sets[token_len].add(token)

    def __del__(self):
        if hasattr(self, 'f') and not self.f.closed:
            self.f.close()

    def _get_file_handle(self):
        if not hasattr(self, 'f') or self.f.closed:
            self.f = open(self.file_path, 'rb')
        return self.f

    def __len__(self):
        return self.total_lines

    def trans_t2id(self, one_str):
        id_list = []
        idx = 0
        max_len = 4
        str_len = len(one_str)

        token_sets = self.token_sets
        copy_token2idx = self.token2idx
        block_len = BLOCK_SIZE + 1
        while idx < str_len and len(id_list) < block_len:
            matched = False
            # 计算当前可匹配的最大长度（避免越界），只计算一次
            current_max_possible_len = min(max_len, str_len - idx)
            # 从最长到最短匹配，直接查对应长度的token集合
            for current_len in range(current_max_possible_len, 0, -1):
                sub_str = one_str[idx:idx + current_len]
                if sub_str in token_sets[current_len]:
                    id_list.append(copy_token2idx[sub_str])
                    idx += current_len
                    matched = True
                    break
            # 未匹配则添加UNK_ID，指针后移1位
            if not matched:
                id_list.append(UNK_ID)
                idx += 1

        pad_num = max(0, block_len - len(id_list))
        if pad_num > 0: id_list += [PAD_ID] * pad_num

        return id_list, one_str[0:idx]

    def __getitem__(self, idx):
        # 跳转到指定行的字节位置，读取该行
        f = self._get_file_handle()
        f.seek(self.line_pos[idx])  # 直接跳转，不用从头读
        line = self.f.readline().decode('utf-8').strip()

        # 解析JSON（标准JSON，直接load即可）
        data = json.loads(line)
        content = data["content"]
        one_str = self.pattern.sub('', content)
        id_list, matched_str = self.trans_t2id(one_str)
        id_tensor = torch.tensor(id_list, dtype=torch.long)
        return id_tensor, self.mask.clone(), matched_str
"""

if __name__ == '__main__':

    d = process_sft_data()
    for src, tgt in d:
        print(src)
        print(tgt)
        print(f'shape: {src.shape}|{tgt.shape}')
        break

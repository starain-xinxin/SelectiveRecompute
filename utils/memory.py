import torch
import logging

class MemoryMonitor:
    def __init__(self, folder: str, name: str = 'MemoryMonitor', device: str='cuda:0', is_snapshot=False):
        self.folder = folder
        self.name = name
        self.save_filename = folder + '/' + name + '.pickle'    # 文件存储
        self.device = device
        self.is_start = False
        self.max_tensor_memory = 0
        self.max_torch_cache = 0
        self.is_snapshot = is_snapshot

    def start(self, max_entries=100000):
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        self.is_start = True
        if self.is_snapshot:
            torch.cuda.memory._record_memory_history(max_entries=max_entries, device=self.device)
        logging.info(f'start the memory monitor of "{self.name}"   device:{self.device}')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=self.device)

    def end(self):
        if self.is_start:
            self.is_start = False
            if self.is_snapshot:
                torch.cuda.memory._dump_snapshot(self.save_filename)    # 保存数据
                torch.cuda.memory._record_memory_history(enabled=None)  # 停掉记录，关闭snapshot
            self.max_tensor_memory = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024
            logging.info(f'张量占用最大显存：{self.max_tensor_memory}MB')
            self.max_torch_cache = torch.cuda.max_memory_reserved(device=self.device) / 1024 / 1024
            logging.info(f'峰值缓存：{torch.cuda.max_memory_reserved(device=self.device) / 1024 / 1024}MB')
            logging.info(f'end the memory monitor of "{self.name}"')
        else:
            logging.error(f'MemoryMonitor "{self.name}": memory is not started')

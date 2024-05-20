import torch
from torch import nn
from datetime import datetime
from torch.autograd.profiler import record_function
import torch.utils.checkpoint as cp


def trace_handler(prof: torch.profiler.profile):
   # 获取时间用于文件命名
   timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
   file_name = f"visual_mem_{timestamp}"

   # 导出tracing格式的profiling
   prof.export_chrome_trace(f"{file_name}.json")

   # 导出mem消耗可视化数据
   prof.export_memory_timeline(f"{file_name}.html", device="cuda:0")


class FFN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.fnn1 = nn.Linear(512,512).to(device=device)
        self.fnn2 = nn.Linear(512,512).to(device=device)
        self.fnn3 = nn.Linear(512,512).to(device=device)
        self.fnn4 = nn.Linear(512,512).to(device=device)
    def forward(self, x):
        x = self.fnn1(x)
        x = self.fnn2(x)
        x = self.fnn3(x)
        x = self.fnn4(x)
        return x

class my_model(nn.Module):
    def __init__(self, device, save_mem):
        super().__init__()
        self.FNN1 = FFN(device)
        self.FNN2 = FFN(device)
        self.tf = nn.Transformer(d_model=512, nhead=2, num_encoder_layers=2, num_decoder_layers=2).to(device=device)
        self.save_mem = save_mem

    def forward(self, x, tgt):
        if self.save_mem:
            x = x+torch.Tensor([0]).cuda()
            tgt = tgt+torch.Tensor([0]).cuda()
            x = cp.checkpoint(self.FNN1, x)
            tgt = cp.checkpoint(self.FNN2, tgt)
            x = cp.checkpoint(self.tf, x, tgt)
        else:
            x = x+torch.Tensor([0]).cuda()
            tgt = tgt+torch.Tensor([0]).cuda()
            x = self.FNN1(x)
            tgt = self.FNN2(tgt)
            x = self.tf(x, tgt)
        return x



def train(save_mem ,num_iter=5, device="cuda:0"):
    model = nn.Transformer(d_model=512, nhead=2, num_encoder_layers=2, num_decoder_layers=2).to(device=device)
    model = my_model(save_mem = save_mem, device=device)
    x = torch.randn(size=(1, 1024, 512), device=device)
    tgt = torch.rand(size=(1, 1024, 512), device=device)
    model.train()
    labels = torch.rand_like(model(x, tgt))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=trace_handler,
    ) as prof:
        for _ in range(num_iter):
            prof.step()
            with record_function("## forward ##"): 
                y = model(x, tgt)

            with record_function("## backward ##"):
                loss = criterion(y, labels)
                loss.backward()
                print(loss.item())

            with record_function("## optimizer ##"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # warm-up:
    train(True, 1)
    # run:
    train(True, 3)
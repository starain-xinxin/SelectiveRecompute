import utils
import torch
import logging
import time
logging.basicConfig(level=logging.INFO)

""" memory monitor """
mem_monitor = utils.MemoryMonitor(folder='/home/yuanxinyu/SelectiveRecompute/data/test', name='test')
mem_monitor.start()

a = torch.tensor([1.,2.,3.]).cuda()
a.requires_grad_()

mem_monitor.end()

""" timer """
timer = utils.Timer('test')
timer.start()

time.sleep(2)

timer.end()
print(timer.runtime())


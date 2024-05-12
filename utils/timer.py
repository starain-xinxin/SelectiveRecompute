import logging
import time

class Timer:
    def __init__(self, name:str='timer'):
        self.name = name
        self.is_start = False
        self.start_time = None      # 开始计时的时间
        self.end_time = None        # 停止计时的时间
        self.run_time = None         # 上一次的计时总时间

    def start(self):
        if self.is_start:
            logging.warning(f'there is a running timer killed -- Timer "{self.name}"')
        self.is_start = True
        self.start_time = time.time()
        self.end_time = None
        logging.info(f'start the timer of "{self.name}"')

    def end(self):
        self.end_time = time.time()
        self.run_time = self.end_time - self.start_time
        self.is_start = False
        logging.info(f'end the timer of "{self.name}", spend {self.run_time} sec')
        return self.run_time

    def runtime(self):
        return self.run_time



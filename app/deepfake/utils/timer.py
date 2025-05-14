# coding: utf-8

"""
tools to measure elapsed time
"""

import time
import sys

class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.max_time = 0.
        self.min_time = sys.float_info.max

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.max_time = max(self.max_time, self.diff) 
        self.min_time = min(self.min_time, self.diff)

        return self.diff

    def clear(self):
        self.start_time = 0.
        self.diff = 0.
        self.calls = 0
        self.max_time = 0.
        self.min_time = sys.float_info.max

    def show(self, label):
        averate = self.total_time / self.calls
        print(f'{label}, total: {self.total_time:.3f}s, average: {averate:.3f}s, max: {self.max_time:.3f}s, min: {self.min_time:.3f} for calls {self.calls} times')
# coding: utf-8

"""
tools to measure elapsed time
"""

import time

class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        return self.diff

    def clear(self):
        self.start_time = 0.
        self.diff = 0.

    def show(self):
        print(f'total time: {self.total_time:.3f}s, for calls {self.calls} times')
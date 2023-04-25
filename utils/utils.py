import os

import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch


def save_model(model, save_path, model_name, epoch):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, model_name + '-{}.pth'.format(epoch))
    torch.save(model.state_dict(), filename)


class Smoother():
    def __init__(self, window):
        self.window = window
        self.num = {}
        self.sum = {}
    def update(self, **kwargs):
        """
        为了调用方便一致，支持kwargs中有值为None的，会被忽略
        kwargs中一些值甚至可以为dict，也就是再套一层。
        示例: update(a=1, b=2, c={'c':1, 'd':3})，相当于update(a=1, b=2, c=1, d=3)
        如果值为参数的None的话忽略
        """
        values = {}
        for key in kwargs:
            if isinstance(kwargs[key], dict):
                for x in kwargs[key]:
                    values[x] = kwargs[key][x] #有可能会覆盖，如update(a=1,b={'a':2})
            else:
                values[key] = kwargs[key]
        for key in values:
            if values[key] is None:
                continue
            if key not in self.num:
                self.num[key] = []
                self.sum[key] = 0
            self.num[key].append(values[key])
            self.sum[key] += values[key]

            if len(self.num[key])>self.window:
                self.sum[key] -= self.num[key][-self.window-1]
            if len(self.num[key])>self.window*2:
                self.clear(key)
        pass
    def clear(self, key):
        del self.num[key][:-self.window]
    def value(self, key = None, mean=True):
        if mean:
            if key is None:
                return {key: self.sum[key] / min(len(self.num[key]),self.window) for key in self.num}
            return self.sum[key] / min(len(self.num[key]),self.window)
        if key is None:
            return {key: np.array(self.num[key]) for key in self.num}
        return np.array(self.sum[key])
    def keys(self):
        return self.num.keys()


from torch.optim.lr_scheduler import LambdaLR
import math
from transformers import get_linear_schedule_with_warmup


def get_cos_schedule_with_warmup(optimizer, num_warmup_steps, epochs, lrf, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        epoch = (epochs - num_warmup_steps) // 1
        step = (current_step - num_warmup_steps) % epoch
        return max(
            1E-6, (1 + math.cos(step * math.pi / epoch)) / 2 * (1 - lrf) + lrf  # cosine
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def build_scheduler(lr_decay_func, optimizer, num_warmup_steps, epochs, lrf, gamma=None):
    if lr_decay_func == 'cosine':
        scheduler = get_cos_schedule_with_warmup(optimizer,
                                                 num_warmup_steps=num_warmup_steps,
                                                 epochs=epochs,
                                                 lrf=lrf)
    elif lr_decay_func == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps,
                                                    epochs)
    return scheduler


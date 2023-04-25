import torch


def build_optimizer(optimizer_name, pg, lr, weight_decay):
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(pg,
                                      # lr=lr,
                                      betas=(0.9, 0.999),
                                      weight_decay=weight_decay,
                                      eps=1e-8)
    else:
        optimizer = torch.optim.SGD(pg,
                                    lr=lr,
                                    momentum=0.9,
                                    weight_decay=weight_decay)
    return optimizer
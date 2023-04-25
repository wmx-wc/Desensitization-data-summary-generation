import sys

sys.path.append("/home/mixiangw/wmx/code_python/u2net/weights/4/1/video")
from utils.ema import EMA

from utils.evaluate import evaluate


from adversarial_train.fgm import FGM
from data.process_data import precess_data
import argparse
import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.build_dataloader import build_valid_dataloader, build_metric_dataloader, \
    build_my_dataloader, build_regular_dataloader
from models.small_bart import Sep2SepModel
from utils.build_optimizer import build_optimizer
import torch
from opt_pre import *
from utils.utils import save_model


def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    device = torch.device(args.device)
    set_seed(seed=args.seed)
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    model_dir = os.path.join(args.result_dir, 'models')
    if os.path.exists(model_dir) is False:
        os.makedirs(model_dir)
    # 数据预处理
    data = precess_data(args.root, 'train.csv', mode='train')
    n_split = int(len(data) * 0.8)
    train_data = data[: n_split]
    valid_data = data[n_split:]
    print('训练集长度：', len(train_data))
    print('验证集长度：', len(valid_data))

    train_loader = build_my_dataloader(args, data, shuffle=True)
    valid_loader = build_valid_dataloader(args, valid_data)
    metric_loader = build_metric_dataloader(args, sorted(valid_data, key=lambda x: len(x[1])))
    print('Data loaded successfully')
    # 加载模型
    model = Sep2SepModel(args.model_name, args.vocab_size, args.label_smoothing).to(device)
    if args.filename is not None and os.path.exists(args.filename):
        # 加载模型权重
        model.load_state_dict(torch.load(args.filename))
        print(args.filename, 'loaded pre-trained model\n')
    else:
        print('not loaded pre-trained model\n')
    pg = [p for p in model.parameters() if p.requires_grad]
    print('Model loaded successfully')

    optimizer = build_optimizer(args.optimizer_name, pg, args.lr, args.weight_decay)
    print('Optimizer configured successfully')
    # # 初始化
    ema = EMA(model, 0.999)
    ema.register()
    best_metrics = 0.

    # 混合精度训练
    for epoch in range(args.epochs):
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('lr:', cur_lr)
        print('weight decay:', optimizer.state_dict()['param_groups'][0]['weight_decay'])
        train_loss, train_lm_loss, train_mlm_loss = train_one_epoch(train_loader, model, optimizer, epoch,
                                                                    n_accumulate=1, device=device, is_adversial=False,
                                                                    ema=ema)
        valid_loss = valid(valid_loader, model, device,  epoch)

        tb_writer.add_scalar('train loss', train_loss, epoch)
        tb_writer.add_scalar('train lm loss', train_lm_loss, epoch)
        tb_writer.add_scalar('train mlm loss', train_mlm_loss, epoch)
        tb_writer.add_scalar('valid loss', valid_loss, epoch)
        tb_writer.add_scalar('lr', cur_lr, epoch)

        if epoch % 2 == 0:
            metrics = evaluate(model.bart_model, metric_loader, args, device)
            if metrics > best_metrics:
                best_metrics = metrics
            print('epoch', epoch, metrics)
            save_model(model, model_dir, args.model_name, epoch)
    print('best_metrics:', best_metrics)


def train_one_epoch(train_loader, model, optimizer, epoch, n_accumulate, device, is_adversial, ema):
    model.train()
    optimizer.zero_grad()
    num_steps = len(train_loader)
    mean_loss = torch.zeros(1).to(device)
    mean_lm_loss = torch.zeros(1).to(device)
    mean_mlm_loss = torch.zeros(1).to(device)
    # 对抗训练
    fgm = FGM(model) if is_adversial else None
    # 在进程 0 中打印训练进度
    data_loader = tqdm(train_loader, total=num_steps)
    for idx, batch in enumerate(data_loader):
        enc_inputs, enc_attention_mask, dec_inputs, dec_labels, dec_masks = (t.type(torch.LongTensor).to(device) for t in batch)

        loss, lm_loss, mlm_loss = model(enc_inputs, None, enc_attention_mask, dec_inputs, dec_labels, dec_masks)
        loss.backward()
        if is_adversial:
            # 对抗训练
            fgm.attack()  # 在embedding上添加对抗扰动
            loss_adv, _, _ = model(enc_inputs, None, enc_attention_mask, dec_inputs, dec_labels, dec_masks)
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复embedding参数
        mean_loss = (mean_loss * idx + loss.item()) / (idx + 1)  # update mean losses
        mean_lm_loss = (mean_lm_loss * idx + lm_loss.item()) / (idx + 1)  # update mean losses

        # Clip the norm of the gradients to 1
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        data_loader.desc = "[train epoch {}] loss: {:.3f}, lm loss: {:.3f}".format(epoch, mean_loss.item(),
                                                                                   mean_lm_loss.item())
        optimizer.step()
        # 更新模型的副本
        ema.update()
        ema.apply_shadow()
        optimizer.zero_grad()
    return mean_loss, mean_lm_loss, mean_mlm_loss


def valid(valid_loader, model, device, epoch):
    model.eval()
    num_steps = len(valid_loader)
    # 在进程 0 中打印训练进度
    valid_loader = tqdm(valid_loader, total=num_steps)
    mean_loss = torch.zeros(1).to(device)
    for i, batch in enumerate(valid_loader):
        input_ids, masks, dec_inputs, lm_labels, dec_masks = (t.type(torch.LongTensor).to(device) for t in batch)
        loss, _, _ = model(input_ids, masks=masks, decoded_inputs=dec_inputs, lm_labels=lm_labels,
                           dec_masks=dec_masks)
        mean_loss = (mean_loss * i + loss.item()) / (i + 1)  # update mean losses
        valid_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, mean_loss.item())
    return mean_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    preprocess_data_opts(parser)
    model_opts(parser)
    batch_size_opts(parser)
    train_opts(parser)
    optimizer_opts(parser)
    scheduler_opts(parser)
    regularization_opts(parser)

    parser.add_argument('--seed', default=7, help='')

    parser.add_argument('--vocab_size', default=1400, help='')

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    parser.add_argument('--is_need_frozen', default=False, type=int, help='Do you need to freeze parameters?: The'
                                                                          ' embedding layer does not need to be frozen')
    parser.add_argument('--num_beams', type=int, default=2)

    opt = parser.parse_args()
    opt.filename = '/home/mixiangw/wmx/code_python/u2net/weights/4/1/video/fune/tune_wd_4_final/models/models/facebook/bart-base-8.pth'
    main(opt)

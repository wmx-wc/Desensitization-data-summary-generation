import sys
sys.path.append("/home/mixiangw/wmx/code_python/u2net/weights/4/1/videography")
import argparse
import os
from tqdm import tqdm

from data.build_dataloader import build_predict_dataloader
from data.process_data import precess_data
from models.bart import Sep2SepModel
from opt import *
import pandas as pd


def decode(model, test_loader, args, device):
    # see ``examples/summarization/bart/run_eval.py`` for a longer example
    model.eval()
    num_steps = len(test_loader)
    # 在进程 0 中打印训练进度
    data_loader = tqdm(test_loader, total=num_steps)
    data = []
    for i, batch_data in enumerate(data_loader):
        ids, input_ids, masks = batch_data
        # Generate Summary
        summary_ids = model.generate(input_ids.to(device), attention_mask=masks.to(device), num_beams=args.num_beams,
                                     max_length=96, early_stopping=False)
        ids = ids.tolist()
        summary_ids = summary_ids.tolist()
        for id, summary in zip(ids, summary_ids):
            # print(summary)
            end = 2
            while end < len(summary):
                if summary[end] == 1 or summary[end] == 2:
                    break
                end += 1
            summary = summary[2: end]
            data.append([id, ' '.join([str(x) for x in summary])]) # if x != 2 and x != 0 and x != 1 and x != 4

    data = pd.DataFrame(data)
    print(data)
    data.to_csv('./sub/preliminary_b_sub.csv', index=False, header=None)


def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    device = torch.device(args.device)
    # 读取测试集数据
    data = precess_data('/home/mixiangw/wmx/code_python/u2net/weights/4/1/videography/datacsv', 'preliminary_b_test.csv', mode='test')
    print('Successfully read data!')

    # 加载测试集dataloader
    test_loader = build_predict_dataloader(args, data)
    print('Data loaded successfully!')
    # 加载模型
    model = Sep2SepModel(args.model_name, args.vocab_size).to(device)
    print('Model initialization successful!')

    if os.path.exists(args.filename):
        print(args.filename)
        # 加载模型权重
        model.load_state_dict(torch.load(args.filename))
        print('loaded pre-trained model\n')
    else:
        print('not loaded pre-trained model\n')
    # 解码
    print('Start decoding!!!')
    decode(model.bart_model, test_loader, args, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    preprocess_data_opts(parser)
    train_opts(parser)
    model_opts(parser)

    parser.add_argument('--vocab_size', default=1400, help='')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--filename', default='/home/mixiangw/wmx/code_python/u2net/weights/4/1/video/fune/fune/models/models/facebook/bart-base-2.pth',
                        help='')

    parser.add_argument('--num_beams', default=2, help='')
    opt = parser.parse_args()
    opt.batch_size = 64
    main(opt)


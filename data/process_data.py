import os
import re

import pandas as pd
from transformers import BartTokenizer


def precess_data(root: str, file_path: str, mode='train'):
    file = os.path.join(root, file_path)
    # 读取数据集
    df = pd.read_csv(file, header=None, encoding='utf-8')
    # 输入数据的最大长度
    max_len = 0
    data = []
    my_dict_ = set()
    data_str = ''
    res = []
    for i in range(len(df)):
        lst = [df[0][i]]
        # 医学影像描述
        description = re.sub(' +', ' ', df[1][i].strip())
        res.append(description)
        description = [int(x) for x in description.split()]
        for i in range(len(description)):
            my_dict_.add(description[i])
        lst.append(description)
        max_len = max(max_len, len(description))
        # 医学影像诊断报告
        if mode == 'train':
            report = re.sub(' +', ' ', df[2][i].strip())
            res.append(report)
            report = [int(x) for x in report.split()]
            lst.append(report)
            for i in range(len(report)):
                my_dict_.add(report[i])
        data.append(lst)
    # lst = list(my_dict_)
    # lst.sort()
    # print(lst)
    # with open('txt/valid_data_b.txt', 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(res))
    print('inputs 最大长度为：', max_len)
    return data


if __name__ == '__main__':
    data = precess_data('E:/Documents/影像学NLP/train', 'preliminary_b_test.csv', mode='test')
    BartTokenizer
    # for item in data:
    #     print(item)

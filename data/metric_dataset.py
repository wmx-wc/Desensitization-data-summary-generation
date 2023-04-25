import torch
from torch.utils.data import Dataset


class MetricDataset(Dataset):
    def __init__(self, meta_data: list, encode_max_length=160, decode_max_length=96):
        self.meta_data = meta_data
        # 序列的最大长度
        self.enc_max_length = encode_max_length
        # 影像医学描述
        self.inputs = []
        self.attention_masks = []
        self.labels = []
        for i in range(len(meta_data)):
            description = meta_data[i][1]
            report = [str(x) for x in meta_data[i][2]]
            self.inputs.append(description)
            self.attention_masks.append([1] * len(description))
            self.labels.append(' '.join(report))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs = torch.tensor(self.inputs[index], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_masks[index], dtype=torch.long)
        return inputs, attention_mask, self.labels[index]
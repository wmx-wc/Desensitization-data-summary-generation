from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from transformers import BartTokenizer

# 读取脱敏后的数据
with open("txt/data.txt", "r", encoding="utf-8") as f:
    data = f.read()

data = data.split('\n')
new_data = []

import ast

# 读取 txt 文件中的字符串，然后将其转换为字典对象
with open('json.txt', 'r', encoding='utf-8') as f:
    my_dict_str = f.read()
    my_dict = ast.literal_eval(my_dict_str)

# 将数字代替的文本转换回来
for i in range(len(data)):
    lst = data[i].split()
    for j in range(len(lst)):
        lst[j] = my_dict[lst[j]]
    new_data.append(''.join(lst))
new_data = '\n'.join(new_data)
print(new_data)


# 初始化 tokenizer
tokenizer = Tokenizer(models.BPE())

# 设置词汇表大小和合并操作的次数
tokenizer.model.train(
    files=[new_data],
    vocab_size=1500,
    min_frequency=2,
    show_progress=True,
)


# 将分词后的结果保存到文件中
with open("vocab.txt", "w", encoding="utf-8") as f:
    f.write(tokenizer.get_vocab())

# 加载保存的分词器
tokenizer_bart = BartTokenizer(vocab_file="vocab.txt")

# 使用加载的分词器进行编码
encoded_data = tokenizer_bart.encode(data).input_ids

print(encoded_data)

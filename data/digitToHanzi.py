import json
import string

my_dict = {}

f = open("E:/Documents/percent/data/book_review.txt", encoding='utf-8')
lines = f.read().strip().replace('\n', '')
f.close()
begin = 9
i = 0
while begin < 1300:
    if lines[i] not in my_dict and not lines[i].isdigit() and lines[i] not in string.ascii_letters:
        if 0x4e00 <= ord(lines[i]) <= 0x9fff:
            my_dict[lines[i]] = begin
            begin += 1
    i += 1
print(my_dict)
my_dict = {value: key for key, value in my_dict.items()}

# 将字典转换成JSON字符串
json_str = json.dumps(my_dict, ensure_ascii=False)

# 打开文件，将JSON字符串写入文件
with open('json.txt', 'w', encoding='utf-8') as f:
    f.write(json_str)


import ast

# 读取 txt 文件中的字符串，然后将其转换为字典对象
with open('json.txt', 'r', encoding='utf-8') as f:
    my_dict_str = f.read()
    my_dict = ast.literal_eval(my_dict_str)


# 输出字典对象
print(my_dict)


f = open("./txt/data.txt", encoding='utf-8')
lines = f.read().split('\n')
data = []

for line in lines:
    sts = line.split()
    new_str = ''
    for st in sts:
        new_str += my_dict[st]
    data.append(new_str)

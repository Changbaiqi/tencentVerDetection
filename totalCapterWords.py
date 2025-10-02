import os
# 用于统计爬取的验证码的字有哪些
# 指定文件夹路径
folder_path = r"E:\Yatori-Dev\tencentImg"  # 修改为你的路径

# 获取文件夹下所有文件名
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 保存所有第一部分的字符
all_chars = []

for name in file_names:
    first_part = name.split('_')[0]  # 用 '_' 分割取第一部分
    all_chars.extend(list(first_part))  # 拆成字符

# 去重并保持顺序
seen = set()
unique_chars_list = []
for c in all_chars:
    if c not in seen:
        seen.add(c)
        unique_chars_list.append(c)

# 输出去重字符列表
print("去重字符列表：", unique_chars_list)

# 输出纯去重字符串
unique_chars_str = ''.join(unique_chars_list)
print("去重字符串：", unique_chars_str)

import json 
import re
def __read_file(filename: str) -> list:
    """
    将本地数据集读到数据加载器中。

    Args:
        filename (str): 数据集文件名

    Returns:
        [tuple] -> 文本列表，标签列表
    Returns:
        [list] -> 文本列表
    """

    data = []  # 存储提取的文本数据
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            json_data = json.loads(line)
            text = json_data.get("text")
            data.append(text)
    return data

# def remove_special_characters(text):
#     # 使用正则表达式去除特殊字符和标点符号
#     cleaned_text = re.sub('[^\w\s]', '', text)
#     return cleaned_text

# # 示例文本
# text = "Hello, world! How are you?"

# # 去除特殊字符和标点符号
# cleaned_text = remove_special_characters(text)
# print(cleaned_text)

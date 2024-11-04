import json
import os
from typing import Any, Dict, List, Union
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

# 自定义Dataset类
class MyDataset(Dataset):
    def __init__(self, raw_data):
        self.data = data_to_list(raw_data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def read_json_file(fpath: Union[str, os.PathLike[str]]) -> Any:
    """Load dictionary from JSON file.

    Args:
        fpath: Path to JSON file.

    Returns:
        Deserialized Python List.
    """
    with open(fpath, "rb") as f:
        return json.load(f)


def save_json_dict(
    json_fpath: Union[str, os.PathLike[str]],
    dictionary: Union[Dict[Any, Any], List[Any]],
) -> None:
    """Save a Python dictionary to a JSON file.

    Args:
        json_fpath: Path to file to create.
        dictionary: Python dictionary to be serialized.
    """
    with open(json_fpath, "w") as f:
        json.dump(dictionary, f)


def data_to_list(data: list) -> list:
    """处理Json文件数据，格式为：Q=[q, [d1, d2, d3, d4...], [label_1, label_2...]].

    Args:
        data: 从json.load读出来的初始数据

    Returns:
        Q: [q, [d1, d2, d3, d4...], [label_1, label_2...]]
            q: 为公共问题
            d1: 为列表，存储着数据
            label_1：对应d1的label
    """
    df_data = pd.DataFrame(data)
    df_data.set_index('问题编号', inplace=True)
    df_data.fillna("-", inplace=True) #把主要成分为null的剔除
    data = df_data.replace('', '-')#把 ”“ 替换为”—“
    choose_cols = ['适应症', '禁忌症', '注意事项', '药物相互作用', '主要成分','用法用量','儿童禁忌', '孕妇禁忌', '老年人禁忌','存储方式']
    #'通用名称','商品名称'
    Q = []
    for i in np.unique(data.index.to_numpy()):
        nums = []
        nums.append(i)
        a = data.loc[i]['问题']

        if isinstance(a, str):
            nums.append(a)
            result = []
            b = []
            for _, row in data.loc[i].iterrows():
                for col in choose_cols:
                           
                    if row[col] != '-':
                        b.append(int(row['应召回']))
                        result.append(f'{row["通用名称"]} {row["商品名称"]} {row["科室分类"]} {col} {row[col]}')
            nums.append(result)
            nums.append(b)
        else:
            # print(a.iloc[0])
            nums.append(a.iloc[0])
            result = []
            b = []
            for _, row in data.loc[i].iterrows():
                for col in choose_cols:
                    if row[col] != '-':
                        b.append(int(row['应召回']))
                        result.append(f'{row["通用名称"]} {row["商品名称"]} {row["科室分类"]} {col} {row[col]}')
            nums.append(result)
            nums.append(b)
        Q.append(nums)
    # print(Q[0])
    return Q


if __name__ == "__main__":
    # 读取初始数据
    raw_data = read_json_file("data/test_demo.json")
    # 创建自定义Dataset实例,并在init过程处理数据
    my_dataset = MyDataset(raw_data)
    # 使用DataLoader批量加载数据
    batch_size = 1
    dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False)
    # 遍历DataLoader并获取数据
    
    for i,batch_data in enumerate(dataloader):
        count_1 =0
        public_q = batch_data[1][0]
        data, label = batch_data[2:]
        n = batch_data[0][0]
        print("Question Number:", n)
        print("Public Question:", public_q)
        print("Batch Data Shape:", len(data))
        print("Batch Label:",len(label))

    

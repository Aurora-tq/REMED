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
        self.data = raw_data

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
    # for i in range(len(data)):
    #     Q = [result[i]["query"], result[i]["content"], result[i]["label"]]
    df_data = pd.DataFrame(data)
    df_data.set_index('问题编号', inplace=True)
    df_data.fillna("-", inplace=True) #把主要成分为null的剔除
    data = df_data.replace('', '-')#把 ”“ 替换为”—“
    # data = data.replace('暂无数据', '-')
    # data = data.replace('尚不明确', '-')
    # data = data.replace('尚不明确。', '-')
    # print(df_data) # 有空值如何处理？
    # print(df_data.index.to_numpy())
    # print(df_data)
    # for col in ["儿童禁忌", "孕妇禁忌", "老年人禁忌", "适应症", "禁忌症", "用法用量", "不良反应",
    #                         "注意事项", "药物相互作用", "主要成分"]:
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
    # print(Q)
    return Q


if __name__ == "__main__":
    # 读取初始数据
    raw_data = read_json_file("/mnt/workspace/pangtianqi/medical_kb_chatbot/train/m3e_train_finetune.json")#m3e_train_finetune dataset2_train

    # 创建自定义Dataset实例,并在init过程处理数据
    my_dataset = MyDataset(raw_data)

    # 使用DataLoader批量加载数据
    batch_size = 1
    dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False)
    sum_label = 0
    sum_data = 0
    labels_list = []
    labels =[]
    cnt =0 
    cit =0
    vector = []
    # 遍历DataLoader并获取数据
    for batch_data in dataloader:
        public_q = batch_data[0]

        data = batch_data[1]
        label = batch_data[2]
        # n = batch_data[0][0]
        # for i in label:
        #     if i == 1:
        #         print("hello")
        #         cnt +=1
        #     elif i == 0:
        #         print("no")
        #         cit +=1
        # print("i==1 cnt",cnt)
        # print("i==0 cnt",cit)
        # for content in data:
        #     print("content",content)
        # print("Question Number:", n)
        print("Public Question:", public_q)
        print("Batch Data Shape:", len(data))
        # print(data)
        print("Batch Label:",len(label))
        # lst_ = []
        # for i in range(0,10):
        #     lst_.append(data[i])
        # print("lst_",lst_)
        # for i in label:
    #     labels_list.append(label)
    # # print(labels_list)
    # vectors = [torch.cat(tensor).tolist() for tensor in labels_list]
    # #value_list = [tensor.item() for tensor in labels_list]
    # print(vectors)
    
    # for batch_data in dataloader:
    
    # for sublist in vectors:
    #     count_0 = 0
    #     count_1 = 0
    #     for value in sublist:
    #         if value == 0:
    #             count_0 += 1
                
    #         elif value == 1:
    #             count_1 += 1
    # print("Count of 0:", count_0)       
    # print("Count of 1:", count_1)
    # k = 10
    # count_all = 20 # 数据库大小
    # count_test = len(value_list)
    # baseline= []
    # ours = []
    # tp = 0
    # l1 = 0
    # precision_all, recall_all, accuracy_all = 0, 0, 0
    # I_ = [190,815,189,350,809,991,1036,353,290,193]
    # for i in I_:
    #     if value_list[i] == 0:
    #         break
    #     else:
    #         l1 = l1+1
    #         if value_list[i] == 1: # 若labels=1的在模型中的label也等于１
    #             tp = tp+1
    # print("tp",tp)
    # print("l1",l1)
        # fp = k-tp
        # fn = l1-tp
        # tn = 50-k-fn

    #     precision = float(tp/(tp+fp))
    #     recall = float(tp/(tp+fn))
    #     accuracy = float((tp+tn)/(tp+fp+tn+fn))

    #     precision_all += precision
    #     recall_all += recall
    #     accuracy_all += accuracy

    # print(f"precision_ave: {float(precision_all/count_test)}")
    # print(f"recall_ave: {float(recall_all/count_test)}")
    # print(f"accuracy_ave: {float(accuracy_all/count_test)}")
    # vectors = [torch.cat(tensor).tolist() for tensor in labels_list]
    # vector.append(vectors)
    # print(vector)
#     zero_count = 0
#     one_count = 0
#     for sublist in vectors:
#         for value in sublist:
#             if value == 0:
#                 zero_count += 1
#             elif value == 1:
#                 one_count += 1
    
#     print("0 的个数:", zero_count)
#     print("1 的个数:", one_count)
   
    # print(len(labels_list))
    # 
    # length = max(len(sub_list) for sub_list in vectors)
    # for i in range(len(vectors)):
    #     sum = 0
    #         # sim_true = 0 # 初始化sim_true
    #         # sim_false = 0
    #     for j in vectors:
    #         if j == 0:  # 负样本
    #             print("hello")
    #         if j== 1:
    #             print("no")
    # print("over")
    # print(length)  
    # for i in vectors:
    #     if vectors[i]==0:
    #         print("hello")
    #     elif  vectors[i]==1 :
    #         print("no")
    # for vector in vectors:
    #     for i in vector:
    #     # 进行标签值的处理或操作
    #         print(i)
    #     sum_label += len(label)
    #     sum_data +=len(data)
    # print(sum_label)
    # print(sum_data)
    # print(len(labels_list))
    

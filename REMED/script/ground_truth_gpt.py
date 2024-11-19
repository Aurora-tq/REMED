import os
os.environ['OPENAI_API_BASE'] = " put your openai key here"

import openai
import pandas as pd
import time
import json
import numpy as np
from train.Dataset_improved import *


class ChatGPTRobot(object):
    def __init__(self):
        pass

    def dialog(self, apiKey, question):
        messages = [
            {"role": "system","content": "假设你是一个具有医学知识的评判官,请综合考虑下面给出的答案与问题之间药品属性和药品名称的语义一致性、信息匹配度和逻辑关联性。Let's think step by step,最终判断若匹配则只需输出1,否则输出0,不需将分析结果进行输出"},
            {"role": "user", "content": question}
        ]
        openai.api_key = apiKey
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                n=1
            )
            reply = ''
            for choice in response.choices:
                reply += choice.message.content
            return reply
        except Exception as e:
            print('error!!', apiKey, e)
            return ''

api_key = 'put your openai key here'
robot = ChatGPTRobot()

# Extract the generated questions from the API response
questions = [choice['text'].strip() for choice in response.choices]

# Print the generated questions
for i, question in enumerate(questions):
  print(f"Question {i+1}: {question}")

# 读取初始数据
raw_data = read_json_file("data/dataset.json")

# 创建自定义Dataset实例,并在init过程处理数据
my_dataset = MyDataset(raw_data)

# 使用DataLoader批量加载数据
batch_size = 1
dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False)

# 遍历DataLoader并获取数据
Labels = []
for batch_data in dataloader:
    # 得到query
    public_q = batch_data[1][0]
    # print(public_q)

    # 提取出answers
    answers = batch_data[2]
    # print(answers)

    # 遍历每一个answer
    labels = []
    for ans in answers:
        ans = ans[0]
        # 用gpt判断其与问题是否匹配
        question = "问题:" + public_q +"\n回答:" + ans
        print(f"question: {question}")
        response = robot.dialog(api_key, question)
        print(response)
        ones_count = response.count('1')
        zeros_count = response.count('0')
        if ones_count > zeros_count:
            l = 1
        else:
            l = 0
        print(f"label: {l}")
        labels.append(l)

    Labels.append(labels)
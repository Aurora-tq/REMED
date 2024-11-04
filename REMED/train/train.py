import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import (
AutoModelForCausalLM,
AutoTokenizer,
DataCollator,
DataCollatorForLanguageModeling,
PreTrainedModel,
PreTrainedTokenizerBase,
Trainer,
TrainingArguments,
)
from os.path import join
from transformers.utils import logging
import json
import time
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from data.data_process.Dataset_improved import *
# logger = logging.get_logger(__name__)
class MychatglmTrain:
    def __init__(
        self, 
        model,
        device = None,
        #args = None,
        # tokenizer = None,
        # data_collator = None
        ):                                                                                                   
            if device is None:
                device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.device = device
            self.model = model

    def Model_score(self,ts1, ts2): 
    # 归一化向量
        normalized_ts1 = ts1 / ts1.norm(dim=0)
        normalized_ts2 = ts2 / ts2.norm(dim=0)
    # 计算余弦相似度
        cos_sim = torch.cosine_similarity(normalized_ts1,normalized_ts2, dim=0)
        return cos_sim

    def Model_loss(self,labels,scores): # labels表示这批数据的label,data_output表示
        sim_false = 0
        sim_true = 0
        for i,label in enumerate(labels):
            if label == 1:
                sim_true += torch.exp(scores[i])
            elif label == 0:
                sim_false += torch.exp(scores[i])
        if sim_false == 0 or sim_true == 0:
            loss = None 
        else:
            loss = -torch.log(sim_true/(sim_true + sim_false))
        return loss
    
    def save(epoch): # 每次完成一个epoch保存一次
        root = "modelset"
        states = {
            'epoch': epoch,
            'model': model.state_dict()
        }
        if not os.path.exists(root):
            os.makedirs(root)
        savepath = join(root, f'model_{epoch+1}.ph')
        torch.save(stats,savepath)
        print(f"saving checkpoint in {savepath}")

    def train(self,model,batch_size=1,epochs=50):
        start_time = time.time()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()) ,lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1.0,gamma=0.95)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        raw_data = read_json_file("/mnt/workspace/pangtianqi/medical_kb_chatbot/data/train_demo.json")
        training_dataset = MyDataset(raw_data)
        training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=True)
        best_loss = float("inf")
        best_model = None
        for epoch in range(epochs):
            s_t = time.time()
            running_loss = 0
            total_train_step = 0
            for batch_data in training_dataloader:
                scores = []#初始化
                model.train()
                query = batch_data[1][0]
                data,labels = batch_data[2:]
                query_embedding = model(query)
                for idx in data:
                    for content in idx:
                        content_embedding = model(content)
                        score = self.Model_score(query_embedding,content_embedding)
                        scores.append(score)
                loss = self.Model_loss(labels,scores)
                if loss is not None:
                    optimizer.zero_grad()
                    loss.backward()#retain_graph=True这个参数不能加
                    torch.nn.utils.clip_grad_norm_(model.parameters(),0.5) # 防止梯度爆炸
                    optimizer.step()
                    total_train_step = total_train_step + 1
                    running_loss += loss.item()
            print("running_loss",running_loss)
            e_t = time.time()
            writer.add_scalar("train_loss",running_loss,global_step = total_train_step)
            print(f"epoch: {epoch+1}, Loss: {running_loss/len(training_dataloader)}, Time: {e_t-s_t}")


            if running_loss < best_loss:
                best_loss = running_loss
                best_model = model
            scheduler.step()

        writer.close()
        end_time = time.time()
        print("total time: ", end_time-start_time)
        # 保存最好的model
        root = "model"
        if not os.path.exists(root):
            os.makedirs(root)
        savepath = join(root, f'best_chatglm2_v1.ph')
        torch.save(best_model.state_dict(), savepath)
        print(f"saving checkpoint in {savepath}")


        
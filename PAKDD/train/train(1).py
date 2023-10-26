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
# from dataloader import kbDataset
from Dataset_improved import *
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
            # if args is None:
            #     output_dir = "tmp_trainer"
            #     logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            #     args = TrainingArguments(output_dir=output_dir)#这里还可以加
            # self.args = args

    def Model_score(self,ts1, ts2): # 求２个tensor之间的相似度
    # 先暂且使用cos来代替，后面可以优化用faiss来代替
    # 归一化向量
        normalized_ts1 = ts1 / ts1.norm(dim=0)
        normalized_ts2 = ts2 / ts2.norm(dim=0)
    # 计算余弦相似度
        cos_sim = torch.cosine_similarity(normalized_ts1,normalized_ts2, dim=0)
        # cos_sim_norm = round(cos_sim,3)
        return cos_sim

    def Model_loss(self,labels,scores): # labels表示这批数据的label,data_output表示
        sim_false = 0
        sim_true = 0
        for i,label in enumerate(labels):
            if label == 1:
                sim_true += torch.exp(scores[i])
            elif label == 0:
                sim_false += torch.exp(scores[i])
        # if sim_true + sim_false!=0:
        # print("sim_true",sim_true)
        # print("sim_false",sim_false)
        if sim_false == 0 or sim_true == 0:
            loss = None 
        else:
            loss = -torch.log(sim_true/(sim_true + sim_false))
            #print("loss",loss)
        # assert cnt != 0
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
        # criterion = nn.BCEWithLogitsLoss()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        raw_data = read_json_file("/mnt/workspace/pangtianqi/medical_kb_chatbot/data/train_demo.json")
        training_dataset = MyDataset(raw_data)
        training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=True)
        # print("here")
        best_loss = float("inf")
        best_model = None
        # embedding_training_dataset.to(device)
        for epoch in range(epochs):
            s_t = time.time()
            running_loss = 0
            total_train_step = 0
            for batch_data in training_dataloader:
                scores = []#初始化
                # loss_total = 0
                model.train()
                query = batch_data[1][0]
                data,labels = batch_data[2:]
                query_embedding = model(query)
                for idx in data:
                    for content in idx:
                        #print("content",content)
                        content_embedding = model(content)
                        score = self.Model_score(query_embedding,content_embedding)
                        scores.append(score)
                # print("scores",scores)
                # print("labels",labels)
                loss = self.Model_loss(labels,scores)
                if loss is not None:
                # print("loss",loss)
                    optimizer.zero_grad()
                    loss.backward()#retain_graph=True这个参数不能加
                    torch.nn.utils.clip_grad_norm_(model.parameters(),0.5) # 防止梯度爆炸
                    optimizer.step()
                    total_train_step = total_train_step + 1
                # print("loss",loss)
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


        
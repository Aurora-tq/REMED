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
from Dataloder import *
import faiss 
import math
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
        #regularization = (self.compute_regularizer(model,query,formatted_data,scores)/len(formatted_data))*1e-4
        for i,label in enumerate(labels):
            if label == 1:
                sim_true += torch.exp(scores[i])
            elif label == 0:
                sim_false += torch.exp(scores[i])
        # if sim_true + sim_false!=0:
        # print("sim_true",sim_true)
        # print("sim_false",sim_false)
        # for j in range(len(lambdas_list)):
        
        if sim_false == 0 or sim_true == 0: #这一行注释掉了
            loss = None 
        else:
            loss = -torch.log(sim_true/(sim_true + sim_false))
        # assert cnt != 0
        return loss

    def compute_regularizer(self,model,query,data):
        #分母为k个最相关的文档的得分
        score_sum = 0 
        encoder = model.embed_model
        content_list_test = []
        query_feat = np.array([encoder.encode(query)])   
        faiss.normalize_L2(query_feat)
        for content in data:
            content_vectors_ = encoder.encode(content)
            content_list_test.append(content_vectors_)
        vectors_test = np.array(content_list_test,dtype=np.float32)
        vectors_test = vectors_test.reshape(len(content_list_test),768)
        # 选择faiss index
        faiss.normalize_L2(vectors_test)
        index_test= faiss.IndexFlatIP(768)
        index_test.train(vectors_test)
        index_test.add(vectors_test)
        D_, I_ = index_test.search(query_feat,int(len(content_list_test)/2)) 
        for i in D_[0]:
             score_sum += i
        lambdas_num = math.exp(score_sum)
        # for i,value in enumerate(scores):
        #regularization = torch.exp(value)/torch.exp(score_sum)
        return lambdas_num
    

    def save(self,epoch): # 每次完成一个epoch保存一次
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
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()) ,lr=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1.0,gamma=0.95)
        # criterion = nn.BCEWithLogitsLoss()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #加载数据集
        vector_data = read_json_file("/mnt/workspace/pangtianqi/medical_kb_chatbot/train/dataset2_RP_train.json")
        # raw_data = read_json_file("/mnt/workspace/pangtianqi/medical_kb_chatbot/data/train_demo.json")
        training_dataset = MyDataset(vector_data)
        training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=True)
        best_loss = float("inf")
        best_model = None
        regularization = 1e-6
        # embedding_training_dataset.to(device)
        for epoch in range(epochs):
            s_t = time.time()
            running_loss = 0
            total_train_step = 0
            regularization = 0
            # 正则项系数，可以根据实际情况调整
            # 计算正则项
            params = model.parameters()  # 获取模型的参数
            regularization_term = 0
            for param in params:
                    regularization_term += torch.norm(param)
            value = regularization_term.item()
            # 使用L2范数作为正则项
            for batch_data in training_dataloader:
                scores = []#初始化
                lambdas_list = []
                # loss_total = 0
                model.train()
                # query = batch_data[0]
                # data = batch_data[1]
                # labels = batch_data[2]
                query = batch_data[0][0]
                # data = batch_data[1]
                data,labels = batch_data[1:]
                # labels = [torch.tensor(value) for value in labels]
                #formatted_data = [tuple(item[0]) for item in batch_data[1]]
                query_embedding = model(query)
                # print(query_embedding.shape)
                #print("query",query)
                #lambdas_num = self.compute_regularizer(model,query,formatted_data)
                #print("lambdas_num",lambdas_num)
                for idx in data:
                    for content in idx:
                        #print("content",content)
                        content_embedding = model(content)
                        #print(content_embedding.shape)
                        score = self.Model_score(query_embedding,content_embedding)
                        scores.append(score)
                        # if lambdas_num != 0 :
                        #lambdas = math.exp(score)/lambdas_num+1
                        #lambdas_list.append(lambdas)
                #regularization += (self.compute_regularizer(model,query,formatted_data,scores)/len(formatted_data))*1e-4
                # print("scores",scores)
                #print("lambdas_list",lambdas_list)
                #lambdas_sum = math.log(sum(lambdas_list))
                #print("lambdas_sum",lambdas_sum)
                # print("regularization",regularization)
                loss = self.Model_loss(labels,scores)
                #print("loss",loss)
                if loss is not None:
                    loss = loss + torch.tensor(regularization * value,requires_grad=True)
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
            print(f"epoch: {epoch+1}, Loss: {running_loss/len(vector_data)}, Time: {e_t-s_t}")


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
        savepath = join(root, f'best_E5_chatglm1_L2_RP_v1.ph')
        torch.save(best_model.state_dict(), savepath)
        print(f"saving checkpoint in {savepath}")


        
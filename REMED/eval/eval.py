import torch
import torchvision
from model.chatglm1_MMD import *
from model.chatglm2_MMD import *
from model.chatglm1_MPD import *
import pandas as pd
import json
import time
import time
import faiss
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from data.data_process.Dataset_improved import *
# from chains.local_doc_qa import LocalDocQA
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.docstore.base import AddableMixin, Docstore
#matplotlib.use("TkAgg")
def _default_relevance_score_fn(score: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        return 1.0 - score / math.sqrt(2)
class MyFAISS(FAISS):
    def __init__(
        self,
        embedding_function: Callable,
        index: Any,
        docstore: Docstore,
        index_to_docstore_id: Dict[int, str],
        relevance_score_fn: Optional[Callable[[float], float]] = _default_relevance_score_fn,
        normalize_L2: bool = True,  # Set the default value to True
    ):
        # Call the parent class (FAISS) constructor
        super().__init__(embedding_function, index, docstore, index_to_docstore_id, relevance_score_fn, normalize_L2)

    def similarity_search_with_score_by_vector(
                self,embedding: List[float], k: int = 4
        ) -> List[Tuple[Document, float]]:
            # query_embedding = np.array([embedding], dtype=np.float32)
            query_embedding = model(query)
            query_feat = query_embedding.cpu()
            query_feat = np.array([query_feat.detach()])
            # query_feat = query_feat.detach().numpy()
            print("query_embedding",query_feat.shape)
            faiss.normalize_L2(query_feat)
            index_type = type(self.index)
            print("索引类型：", index_type)
            # 查看索引维度
            index_dimension = self.index.d
            print("索引维度：", index_dimension)

            scores, indices = self.index.search(query_embedding, k) #self.index.search(np.array([embedding], dtype=np.float32), k)
            #search方法接收查询嵌入和最近邻居数量‘K’作为输入，并返回两个数组
            #indices是一个整数数组，表示与查询嵌入最相似的前‘k'个嵌入向量的索引值。
            print('scores, indices',scores, indices)
            # scores = [cosine_to_similarity(score) for score in scores]
            # print('scores, indices',scores, indices)
            docs = []
            id_set = set()
            store_len = len(self.index_to_docstore_id)
            for j, i in enumerate(indices[0]):
                if i == -1 or 0 < self.score_threshold < scores[0][j]:
                    # This happens when not enough docs are returned.
                    continue
                _id = self.index_to_docstore_id[i]
                doc = self.docstore.search(_id)
                if not self.chunk_content:
                    if not isinstance(doc, Document):
                        raise ValueError(f"Could not find document for id {_id}, got {doc}")
                    doc.metadata["score"] = int(scores[0][j])
                    docs.append(doc)
                    continue
                id_set.add(i)
                docs_len = len(doc.page_content)
                for k in range(1, max(i, store_len - i)):
                    break_flag = False
                    for l in [i + k, i - k]:
                        if 0 <= l < len(self.index_to_docstore_id):
                            _id0 = self.index_to_docstore_id[l]
                            doc0 = self.docstore.search(_id0)
                            if docs_len + len(doc0.page_content) > self.chunk_size:
                                break_flag = True
                                break
                            elif doc0.metadata["source"] == doc.metadata["source"]:
                                docs_len += len(doc0.page_content)
                                id_set.add(l)
                    if break_flag:
                        break

vector_data = read_json_file("/mnt/workspace/pangtianqi/medical_kb_chatbot/dataset.json")
vector_dataset = MyDataset(vector_data)
vector_dataloader = torch.utils.data.DataLoader(vector_dataset, batch_size=1, shuffle=False)


raw_data = read_json_file("/mnt/workspace/pangtianqi/medical_kb_chatbot/test_demo.json")
testing_dataset = MyDataset(raw_data)
testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=False)

# 加载训练好的模型
config = ChatGLMConfig()
model = MyMLP(config)
#model =  MyGLU(hidden_size = 768)
device = model.device
model.load_state_dict(torch.load("model/best_model.ph"))
#model.load_state_dict(torch.load("model/best_chatglm1.ph"))
model = model.to(device)
# print(model)
encoder = model.embed_model
dim =768
testing_text = []
labels_list = []
labels_list_recall = []
for batch_data in testing_dataloader:
    data, label = batch_data[2:]
    for idx in data:
        for content in idx:
            testing_text.append(content)
    labels_list_recall.append(label)
    for i in label:
        labels_list.append(i)
    # print(labels_list)
labels_recall = [torch.cat(tensor).tolist() for tensor in labels_list_recall]
labels = [tensor.item() for tensor in labels_list]
# 将doc进行embed
content_list = []
#embedding vector_store
for batch_data in testing_text:
    # data, label = batch_data[2:]
    content_vectors = encoder.encode(batch_data)
    content_list.append(content_vectors)

vectors = np.array(content_list)
# print("vectors",len(vectors))
# 选择faiss index
faiss.normalize_L2(vectors)
index = faiss.IndexFlatIP(dim)
# 数据插入
# 训练索引获取重构能力
index.train(vectors)
index.add(vectors)
# print("index.ntotal",index.ntotal)

# 优化
# 数据嵌入
# 将doc进行embed
content_list_ = []
#embedding vector_store
for batch_data in testing_text:
    # data, label = batch_data[2:]
    content_vectors_ = model(batch_data)
    content_vectors_ = content_vectors_.cpu()
    content_vectors_ = content_vectors_.detach().numpy()
    content_list_.append(content_vectors_)

vectors_ = np.array(content_list_)
# print("vectors_",len(vectors_))

# 选择faiss index
faiss.normalize_L2(vectors_)
index_ = faiss.IndexFlatIP(dim)
# 数据插入
# 训练索引获取重构能力
index_.train(vectors_)
index_.add(vectors_)
# print("index_.ntotal",index_.ntotal)


# TP - topk中label为１的个数
# FP - topk中label为0的个数
# FN - 不在topk中但label为１的个数
# TN - 不在topk中label为0的个数
# 召回topK
k = 30
count = 0

precision_all, recall_all, accuracy_all = 0, 0, 0
start_time = time.time()
label_pair = []
D_pair = []
score_pair = []
for batch_data in testing_dataloader:
    query = batch_data[1][0]
    # data, label = batch_data[2:]
    # print(len(data))
    query_feat = np.array([encoder.encode(query)]) # 得到baseline query embed
    faiss.normalize_L2(query_feat)
    #query_feat_ = encoder.encode(query) # 得到model query embed
    # query_feat_ = query_feat_.cpu()
    # query_feat_ = np.array([query_feat_.detach()])
    D, I = index.search(query_feat, k) #提取出ours的topk
    D_pair.append(D)
    for i in I[0]:
        label_pair.append(labels[i])
    #print('scores, indices',D, I[0])
    tp, fp, fn, tn = 0, 0, 0, 0
    # 表示真正label是１的个数
    count_0 = 0
    for cnt in labels:
        # if cnt == 1:
        #     l1+=1
        if cnt == 0:
            count_0 +=1
    for sublist in labels_recall:
        # count_0 = 0
        count_1 = 0
        for value in sublist:
            # if value == 0:
            #     count_0 += 1
            if value == 1:
                count_1 += 1
    for i in I[0]:
        if labels[i] == 0:
            break
        elif labels[i] == 1: # 若labels=1的在模型中的label也等于１
            tp = tp+1
    # print("tp",tp)
    fp = k-tp
    # print("fp",fp)
    fn = count_1-tp
    # print("fn",fn)
    tn = count_0-fp
    # print("tn",tn)
    if tp != 0 and fp !=0:
        count +=1
        precision = float(tp/(tp+fp))
        recall = float(tp/(tp+fn))
        accuracy = float((tp+tn)/(tp+fp+tn+fn))

        precision_all += precision
        recall_all += recall
        accuracy_all += accuracy
for item in D_pair:
    for score in item:
        score_pair.extend(score)
print("score_pair",len(score_pair))
print("label_pair",len(label_pair))
print("count",count)
print(f"precision_ave_m3e: {float(precision_all/count)}")
print(f"recall_ave_m3e: {float(recall_all/count)}")
print(f"accuracy_ave_m3e: {float(accuracy_all/count)}")
# content = [item for sublist in data for item in sublist]
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time spent: {elapsed_time:.5f} seconds")

from sklearn.metrics import average_precision_score
mAP = average_precision_score(label_pair, score_pair, average='macro')
print("mAP_m3e:", mAP)

count= 0
precision_all, recall_all, accuracy_all = 0, 0, 0
start_time = time.time()
label_pair_ = []
D_pair_ = []
score_pair_ = []
for batch_data in testing_dataloader:
    query = batch_data[1][0]
    # data, label = batch_data[2:]
    # print(len(data))
    # query_feat = np.array([encoder.encode(query)]) # 得到baseline query embed
    query_feat_ = model(query) # 得到model query embed
    query_feat_ = query_feat_.cpu()
    query_feat_ = np.array([query_feat_.detach()])
    faiss.normalize_L2(query_feat_)
    D_, I_ = index_.search(query_feat_, k) #提取出ours的topk
    # print('scores_, indices_',D_, I_[0])
    D_pair_.append(D_)
    for i in I_[0]:
        label_pair_.append(labels[i])
    # lst = [] # 模型召回的
    tp, fp, fn, tn = 0, 0, 0, 0
    count_0 = 0
    for cnt in labels:
        # if cnt == 1:
        #     l1+=1
        if cnt == 0:
            count_0 +=1
    for sublist in labels_recall:
        # count_0 = 0
        count_1 = 0
        for value in sublist:
            # if value == 0:
            #     count_0 += 1
            if value == 1:
                count_1 += 1
    for i in I_[0]:
        if labels[i] == 0:
            break
        elif labels[i] == 1: # 若labels=1的在模型中的label也等于１
            tp = tp+1
    # print("tp",tp)
    fp = k-tp
    # print("fp",fp)
    fn = count_1-tp
    # print("fn",fn)
    tn = count_0-fp
    
    if tp != 0 and fp !=0: #排除label全为1或全为0的情况
        count+=1
        precision = float(tp/(tp+fp))
        recall = float(tp/(tp+fn))
        accuracy = float((tp+tn)/(tp+fp+tn+fn))

        precision_all += precision
        recall_all += recall
        accuracy_all += accuracy
#print("count",count)
for item in D_pair_:
    for score in item:
        score_pair_.extend(score)
print("score_pair",len(score_pair_))
print("label_pair",len(label_pair_))
print(f"precision_ave_glm2: {float(precision_all/count)}")
print(f"recall_ave_glm2: {float(recall_all/count)}")
print(f"accuracy_ave_glm2: {float(accuracy_all/count)}")
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time spent: {elapsed_time:.5f} seconds")

#保存score_pair和label_pair
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,precision_recall_curve
# 计算不同阈值下的精确率和召回率
precision, recall, thresholds = precision_recall_curve(label_pair_, score_pair_)

# 计算每个阈值下的平均精度均值
# average_precision = average_precision_score(true_labels, predicted_probs)
mAP = average_precision_score(label_pair_, score_pair_, average='macro')
print("mAP_glm1:", mAP)

# 绘制曲线图
plt.plot(recall, precision, color='b', label=f'mAP={mAP:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('mAP Curve')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


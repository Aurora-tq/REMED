import math
import copy
import warnings
import re
import sys

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
from torch.nn import CrossEntropyLoss, LayerNorm
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput
from transformers import PretrainedConfig
from sentence_transformers import SentenceTransformer
from configuration_chatglm import ChatGLMConfig
# from chains.local_doc_qa import LocalDocQA
# from configs.common_config import *
# from chains.local_doc_qa import MyFAISS

def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs

class MyMLP(torch.nn.Module):
    """MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, device=None,embed_model = None,embedding_size = None,layer_norm=None,dropout = None):
        super(MyMLP, self).__init__()
        self.embed_model = embed_model
        self.add_bias = config.add_bias_linear
        # self.embeddings = embeddings
        # self.vector_store = vector_store
        # if vs_path is None:
        #     vs_path = '/mnt/workspace/pangtianqi/medical_kb_chatbot/vector_store/test_1'
        # #Project to 4h.
        # self.vs_path = vs_path
        if device is None:
            device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        # embedding
        self.embed_model = SentenceTransformer('your/path/to/moka-ai/m3e-base', device)
        # self.embed_model = SentenceTransformer('/mnt/workspace/pangtianqi/medical_kb_chatbot/e5-base-v2', device)
        self.embedding_size = self.embed_model.get_sentence_embedding_dimension()
        print(f"Embedding size: {self.embedding_size}")  
        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.layer_norm = LayerNorm((768,),eps=1e-05,elementwise_affine=True)
        self.dropout = nn.Dropout(p=0.1)
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.hidden_size * 8,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu
        # Project back to h.
        #ffn_hidden_size
        self.dense_4h_to_h = nn.Linear(
            config.hidden_size*4,
            config.hidden_size,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )
        
     

    # def init_embedding(self, embedding_model: str = EMBEDDING_MODEL, embedding_device=EMBEDDING_DEVICE, top_k=VECTOR_SEARCH_TOP_K):
    #     self.embed_model = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model], model_kwargs={'device': embedding_device})
    #     self.top_k = top_k
    
    # def init_vector_store(self):
    #     self.vector_store = MyFAISS.load_local(self.vs_path, self.embeddings)    

    def forward(self,data):
            """
            query: [seq_len, batch, hidden_size]
            content: list of [seq_len, batch, hidden_size] or [num_content, seq_len, batch, hidden_size]
            """
            # if self.embed_model is None:
            #     self.init_embedding()

            # if self.vector_store is None:
            #     print('初始化加载向量库')
            #     self.init_vector_store()
        
            embedding = torch.from_numpy(self.embed_model.encode(data))#.to(self.device)
            #embedding.requires_grad = True
            # embedding = self.vector_store.embedding_function(data)
            embedding_tensor = embedding.clone().detach().requires_grad_(True)
            embedding_tensor_cuda = embedding_tensor.to(self.device)
            embedding_tensor_norm = self.layer_norm(embedding_tensor_cuda)#.to(self.device)
            # embedding_tensor_dropout = self.dropout(embedding_tensor_cuda)
            intermediate_parallel = self.dense_h_to_4h(embedding_tensor_norm)
            #暂时不加这里
            intermediate_parallel = self.activation_func(intermediate_parallel)
            #intermediate_parallel = self.dropout(intermediate_parallel)
            output = self.dense_4h_to_h(intermediate_parallel)
            # output_norm = self.layer_norm(output)
            return output

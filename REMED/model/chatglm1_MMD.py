import math
import copy
import os
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

from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput
from sentence_transformers import SentenceTransformer
from configuration_chatglm import ChatGLMConfig

# from chains.local_doc_qa import LocalDocQA
# from configs.common_config import *
# from chains.local_doc_qa import MyFAISS



def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


def gelu(x):
    return gelu_impl(x)


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)

class MyGLU(torch.nn.Module):
    def __init__(
        self, 
        hidden_size, 
        inner_hidden_size=None,
        layer_id=None, 
        bias=True, 
        activation_func=gelu,
        params_dtype=torch.float, 
        empty_init=False,
        embed_model=None,
        vector_store = None,
        layer_norm = None,
        device = None,
        vs_path = None
        ):
        super(MyGLU, self).__init__()
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        self.layer_id = layer_id
        self.activation_func = activation_func
        self.dropout = nn.Dropout(p=0.2)
        # self.vector_store = vector_store
        # self.layer_norm = layer_norm
        if device is None:
            device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.embed_model = SentenceTransformer('your/path/to/moka-ai/m3e-base', device)
        #self.embed_model = SentenceTransformer('/mnt/workspace/pangtianqi/medical_kb_chatbot/e5-base-v2', device)
        embedding_size = self.embed_model.get_sentence_embedding_dimension()
        print(f"Embedding size: {embedding_size}")
        # if vs_path is None:
        #     vs_path = '/mnt/workspace/pangtianqi/medical_kb_chatbot/vector_store/test_1'
        # # Project to 4h.
        # self.vs_path = vs_path
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = init_method(
            torch.nn.Linear,
            self.hidden_size,
            self.inner_hidden_size,
            bias=bias,
            dtype=params_dtype,
        )
        # Project back to h.
        self.dense_4h_to_h = init_method(
            torch.nn.Linear,
            self.inner_hidden_size,
            self.hidden_size,
            bias=bias,
            dtype=params_dtype,
        )
        self.layer_norm = LayerNorm((768,),eps=1e-12,elementwise_affine=True)
        
    # def init_embedding(self, embedding_model: str = EMBEDDING_MODEL, embedding_device=EMBEDDING_DEVICE, top_k=VECTOR_SEARCH_TOP_K):
    #     self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model], model_kwargs={'device': embedding_device})
    #     self.top_k = top_k

    def forward(self,data):
        """
        query: [seq_len, batch, hidden_size]
        content: list of [seq_len, batch, hidden_size] or [num_content, seq_len, batch, hidden_size]
        """
        # if self.embeddings is None:
        #     self.init_embedding()

        # if self.vector_store is None:
        #     print('初始化加载向量库')
        #     self.init_vector_store()
        # print(self.layer_norm.device)
        embedding = torch.from_numpy(self.embed_model.encode(data))#.to(self.device)
        #embedding.requires_grad = True
        # embedding = self.vector_store.embedding_function(data)
        embedding_tensor = embedding.clone().detach().requires_grad_(True)
        # embedding_tensor.to(self.device)
        embedding_tensor_cuda = embedding_tensor.to(self.device)
        #embedding_tensor = torch.tensor(embedding, dtype=torch.float)#.to(self.device)
        embedding_tensor_norm = self.layer_norm(embedding_tensor_cuda)#.to(self.device)
        # embedding_tensor_dropout = self.dropout(embedding_tensor_norm)
        intermediate_parallel = self.dense_h_to_4h(embedding_tensor_norm)
        # intermediate_parallel = self.dropout(intermediate_parallel)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        #output_norm = self.layer_norm(output)
        return output

chatglm1加载:
from chatglm1 import *
model =  MyGLU(hidden_size = 768)
device = model.device
model.load_state_dict(torch.load("model/best_chatglm1.ph"))
model = model.to(device)

chatglm2加载:
config = ChatGLMConfig()
model = MyMLP(config)
device = model.device
model.load_state_dict(torch.load("model/best_chatglm2_v1.ph"))
model = model.to(device)

使用见eval.ipynb
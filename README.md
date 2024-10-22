# REMED
## _Retrieval-Augmented Medical Document Query Responding with Embedding Fine-Tuning_

## DATASET
```sh
MMD- Medical Menu Dataset
MPD- Medical Paper Dataset
```

## MODEL
Modify in test.py
Load chatglm1:
```sh
from chatglm1 import *
model =  MyGLU(hidden_size = 768)
device = model.device
model.load_state_dict(torch.load("model/best_chatglm1.ph"))
model = model.to(device)
```

Load chatglm2：
```sh
from chatglm2 import *
config = ChatGLMConfig()
model = MyMLP(config)
device = model.device
model.load_state_dict(torch.load("model/best_chatglm2_v1.ph"))
model = model.to(device)
```

## EVAL
Bechmark:M3E(Chinese),T5(English)
```sh
python eval.py
```


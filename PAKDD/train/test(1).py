#from train import MychatglmTrain
from finetune_dataset2 import MychatglmTrain
from chatglm2 import *
from chatglm1 import *
# from Dataloder import *
from configuration_chatglm import ChatGLMConfig
if __name__ == "__main__":
    config = ChatGLMConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = MyMLP(config)
    # model.load_state_dict(torch.load("model/best_chatglm2_v1.ph"))
    model = MyGLU(hidden_size = 768)
    # model.load_state_dict(torch.load("model/best_E5_chatglm1_L2.ph"))
    # model.train()
    # print(model)
    model.to(device)
    trainer = MychatglmTrain(model)
    trainer.train(model)


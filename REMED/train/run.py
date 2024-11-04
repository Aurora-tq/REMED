from train import MychatglmTrain
# from finetune.finetune_MPD import MychatglmTrain
from model.chatglm2_MMD import *
from model.chatglm1_MMD import *
from model.chatglm1_MPD import *
# from Dataloder import *
from model.configuration_chatglm import ChatGLMConfig
if __name__ == "__main__":
    config = ChatGLMConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyGLU(hidden_size = 768)
    model.load_state_dict(torch.load("model/best_E5_chatglm1.ph"))
    model.to(device)
    trainer = MychatglmTrain(model)
    trainer.train(model)


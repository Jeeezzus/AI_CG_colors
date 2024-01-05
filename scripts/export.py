import torch
from model import Autoencoder, AutoV2, AutoV2_Lite, Autoencoder_Max, Autoencoder_3, Autoencoder_3_Ultimate
from torch.autograd import Variable


def convert_to_onnx(model_class, model_path, onnx_path, input_shape):
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    dummy_input = Variable(torch.randn(input_shape))
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True)


model_class = Autoencoder_3_Ultimate
model_path = "D:/GithUB/AI_CG_colors/models_checkpoints/auto3_ultimate/32--Autoencoder_3_Ultimate--0.1247.pth"  
onnx_path = "auto3_ultimate.onnx" 
input_shape = (1, 4, 512, 512)

convert_to_onnx(model_class, model_path, onnx_path, input_shape)

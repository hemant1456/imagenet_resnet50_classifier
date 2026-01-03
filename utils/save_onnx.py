import torch
checkpoint = torch.load('./runpod_trained_model.pt',map_location=torch.device('cpu'))

from models.tiny_imagenet_resnet_50 import resnet50

model = resnet50(n_classes=200)

model.load_state_dict(checkpoint['model_params'])

model.eval()


from onnxruntime.quantization import quantize_dynamic, QuantType
onnx_program = torch.onnx.export(
    model = model,
    args = torch.randn(1,3,64,64),
    f = 'final_model_imagenet.onnx',
    input_names = ['input'],
    output_names= ['output'],
    dynamic_axes={
        'input': {0: 'batch_size'}, 
        'output': {0: 'batch_size'}
    },
    opset_version=14,
)


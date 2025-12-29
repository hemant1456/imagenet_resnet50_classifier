import gradio as gr
import numpy as np
from PIL import Image

import onnxruntime as ort
session = ort.InferenceSession('final_model_aws.onnx',providers=['CPUExecutionProvider'])

def predict(image):

    image = Image.fromarray(image).resize((32,32)).convert('RGB')
    image = np.array(image)/255

    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, 3)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 1, 3)
    
    # 4. Apply Normalization
    image = (image - mean) / std

    image = np.moveaxis(image,-1,0)[None,...].astype(np.float32)

    

    output = session.run(output_names=None,input_feed={'input':image})
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    max_idx = int(np.argmax(output[0],axis=1))
    prediction=  classes[max_idx]

    return prediction

demo = gr.Interface(
    fn = predict,
    inputs=gr.Image(),
    outputs = gr.Label(),
    title = "cifar 10 resnet 18 classifier"

)


if __name__=="__main__":
    demo.launch()
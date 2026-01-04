import gradio as gr
import numpy as np
from PIL import Image

import onnxruntime as ort
session = ort.InferenceSession('final_model_imagenet.onnx',providers=['CPUExecutionProvider'])


def load_tiny_imagenet_labels(words_path='./tiny_imagenet_data/tiny-imagenet-200/words.txt', wnids_path='./tiny_imagenet_data/tiny-imagenet-200/wnids.txt'):
    # Load all human-readable names
    wnid_to_name = {}
    with open(words_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                wnid_to_name[parts[0]] = parts[1].split(',')[0] # First common name

    # Load the 200 WNIDs used in Tiny ImageNet
    with open(wnids_path, 'r') as f:
        tiny_wnids = [line.strip() for line in f.readlines()]
    
    # CRITICAL: ImageFolder sorts folders ALPHABETICALLY
    tiny_wnids.sort()
    
    # Create the final list of human names in the correct index order
    return [wnid_to_name.get(wnid, wnid) for wnid in tiny_wnids]

# Load labels once
TINY_LABELS = load_tiny_imagenet_labels()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def predict(image):
    # 1. Resize to match training (64x64)
    img = Image.fromarray(image).convert('RGB').resize((64, 64))
    
    # 2. ToImage() and ToDtype(scale=True) equivalent
    img_array = np.array(img).astype(np.float32) / 255.0

    # 3. Use the EXACT mean/std used in tiny_imagenet_train_transformations
    # Even if they were for CIFAR, they are what the model weights expect.
    mean = np.array([(0.480, 0.448, 0.398)])
    std = np.array([0.276, 0.269, 0.282])
    
    img_array = (img_array - mean) / std

    # 4. Transpose to (C, H, W) and add Batch dimension (1, C, H, W)
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)


    # 6. EXPLICIT CAST TO FLOAT32 (This fixes your error)
    img_array = img_array.astype(np.float32)

    # 5. Inference
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_array})[0]

    # 6. Process Probabilities
    probs = softmax(output)[0]
    
    # Create the dictionary for gr.Label
    confidences = {
        TINY_LABELS[i]: float(probs[i]) 
        for i in range(len(TINY_LABELS))
    }

    return confidences

demo = gr.Interface(
    fn = predict,
    inputs=gr.Image(),
    outputs = gr.Label(num_top_classes=5),
    title = "cifar 10 resnet 18 classifier"

)


if __name__=="__main__":
    demo.launch()
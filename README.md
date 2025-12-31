# CIFAR-10 ResNet-18
Implementation and experimentation with a ResNet-18 inspired architecture on the CIFAR-10 dataset.

The model is live at this url : https://huggingface.co/spaces/hemant-bhambhu/cifar_10_predictor

---

## Architecture and Parameter Size
The following structure represents the ResNet-18 variant used for this project. The model utilizes a series of residual blocks to maintain gradient flow.



```text
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
resnet18                                 [1, 10]                   --
├─Conv2d: 1-1                            [1, 32, 32, 32]           896
├─BatchNorm2d: 1-2                       [1, 32, 32, 32]           64
├─ReLU: 1-3                              [1, 32, 32, 32]           --
├─resnet_block: 1-4                      [1, 64, 16, 16]           --
│    └─basic_block1: 2-1                 [1, 64, 16, 16]           --
│    │    └─Conv2d: 3-1                  [1, 64, 16, 16]           2,048
│    │    └─Conv2d: 3-2                  [1, 64, 16, 16]           18,432
│    │    └─BatchNorm2d: 3-3             [1, 64, 16, 16]           128
│    │    └─ReLU: 3-4                    [1, 64, 16, 16]           --
│    │    └─Conv2d: 3-5                  [1, 64, 16, 16]           36,864
│    │    └─BatchNorm2d: 3-6             [1, 64, 16, 16]           128
│    │    └─ReLU: 3-7                    [1, 64, 16, 16]           --
│    └─basic_block2: 2-2                 [1, 64, 16, 16]           --
│    │    └─Conv2d: 3-8                  [1, 64, 16, 16]           36,864
│    │    └─BatchNorm2d: 3-9             [1, 64, 16, 16]           128
│    │    └─ReLU: 3-10                   [1, 64, 16, 16]           --
│    │    └─Conv2d: 3-11                 [1, 64, 16, 16]           36,864
│    │    └─BatchNorm2d: 3-12            [1, 64, 16, 16]           128
│    │    └─ReLU: 3-13                   [1, 64, 16, 16]           --
├─resnet_block: 1-5                      [1, 128, 8, 8]            --
│    └─basic_block1: 2-3                 [1, 128, 8, 8]            --
│    │    └─Conv2d: 3-14                 [1, 128, 8, 8]            8,192
│    │    └─Conv2d: 3-15                 [1, 128, 8, 8]            73,728
│    │    └─BatchNorm2d: 3-16            [1, 128, 8, 8]            256
│    │    └─ReLU: 3-17                   [1, 128, 8, 8]            --
│    │    └─Conv2d: 3-18                 [1, 128, 8, 8]            147,456
│    │    └─BatchNorm2d: 3-19            [1, 128, 8, 8]            256
│    │    └─ReLU: 3-20                   [1, 128, 8, 8]            --
│    └─basic_block2: 2-4                 [1, 128, 8, 8]            --
│    │    └─Conv2d: 3-21                 [1, 128, 8, 8]            147,456
│    │    └─BatchNorm2d: 3-22            [1, 128, 8, 8]            256
│    │    └─ReLU: 3-23                   [1, 128, 8, 8]            --
│    │    └─Conv2d: 3-24                 [1, 128, 8, 8]            147,456
│    │    └─BatchNorm2d: 3-25            [1, 128, 8, 8]            256
│    │    └─ReLU: 3-26                   [1, 128, 8, 8]            --
├─resnet_block: 1-6                      [1, 256, 4, 4]            --
│    └─basic_block1: 2-5                 [1, 256, 4, 4]            --
│    │    └─Conv2d: 3-27                 [1, 256, 4, 4]            32,768
│    │    └─Conv2d: 3-28                 [1, 256, 4, 4]            294,912
│    │    └─BatchNorm2d: 3-29            [1, 256, 4, 4]            512
│    │    └─ReLU: 3-30                   [1, 256, 4, 4]            --
│    │    └─Conv2d: 3-31                 [1, 256, 4, 4]            589,824
│    │    └─BatchNorm2d: 3-32            [1, 256, 4, 4]            512
│    │    └─ReLU: 3-33                   [1, 256, 4, 4]            --
│    └─basic_block2: 2-6                 [1, 256, 4, 4]            --
│    │    └─Conv2d: 3-34                 [1, 256, 4, 4]            589,824
│    │    └─BatchNorm2d: 3-35            [1, 256, 4, 4]            512
│    │    └─ReLU: 3-36                   [1, 256, 4, 4]            --
│    │    └─Conv2d: 3-37                 [1, 256, 4, 4]            589,824
│    │    └─BatchNorm2d: 3-38            [1, 256, 4, 4]            512
│    │    └─ReLU: 3-39                   [1, 256, 4, 4]            --
├─resnet_block: 1-7                      [1, 512, 2, 2]            --
│    └─basic_block1: 2-7                 [1, 512, 2, 2]            --
│    │    └─Conv2d: 3-40                 [1, 512, 2, 2]            131,072
│    │    └─Conv2d: 3-41                 [1, 512, 2, 2]            1,179,648
│    │    └─BatchNorm2d: 3-42            [1, 512, 2, 2]            1,024
│    │    └─ReLU: 3-43                   [1, 512, 2, 2]            --
│    │    └─Conv2d: 3-44                 [1, 512, 2, 2]            2,359,296
│    │    └─BatchNorm2d: 3-45            [1, 512, 2, 2]            1,024
│    │    └─ReLU: 3-46                   [1, 512, 2, 2]            --
│    └─basic_block2: 2-8                 [1, 512, 2, 2]            --
│    │    └─Conv2d: 3-47                 [1, 512, 2, 2]            2,359,296
│    │    └─BatchNorm2d: 3-48            [1, 512, 2, 2]            1,024
│    │    └─ReLU: 3-49                   [1, 512, 2, 2]            --
│    │    └─Conv2d: 3-50                 [1, 512, 2, 2]            2,359,296
│    │    └─BatchNorm2d: 3-51            [1, 512, 2, 2]            1,024
│    │    └─ReLU: 3-52                   [1, 512, 2, 2]            --
├─AdaptiveAvgPool2d: 1-8                 [1, 512, 1, 1]            --
├─Conv2d: 1-9                            [1, 10, 1, 1]             5,130
==========================================================================================
Total params: 11,154,890
Trainable params: 11,154,890
Non-trainable params: 0
Total mult-adds (M): 135.15
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 2.74
Params size (MB): 44.62
Estimated Total Size (MB): 47.37
==========================================================================================

# Training and Results

The model was trained on the **CIFAR-10** dataset to evaluate the performance of the ResNet-18 architecture. Below are the training parameters and the resulting performance metrics.

---

## 📊 Performance Summary

| Metric | Value |
| :--- | :--- |
| **Epochs** | 10 |
| **Data Augmentation** | None (Default) |
| **Training Accuracy** | **88.6%** |
| **Test Accuracy** | **82.43%** |

---

## 🕒 Training Progress
The following logs represent the final state of the training and testing phases after the 10th epoch.

### Training Log
```bash
epoch = 10: 100%|█████████████████████████| 3125/3125 [00:50<00:00, 62.15it/s, train_loss=0.404, train_accuracy=0.886] test_accuracy = 0.82

After applying following augmentation mixup, randomaffine with p=0.2, horizontal flip p=0.2 and increasing epoch to 30 resulted in test accuracy = 0.883

After adding cutmix augmentation as well along with above augmentation, the test_accuracy marginally improved to 0.887

did aggressive randomaffine and horizontal flip at probability 0.5, accuracy dropeed to 0.878

applied label smoothing while removing above aggeressive augmentation, accuracy reached 0.886
import torchvision
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, AdamW
from tqdm import tqdm
#from models.cifar10_resnet_18 import resnet18
from models.tiny_imagenet_resnet_18 import resnet18
from models.tiny_imagenet_resnet_50 import resnet50
from torchinfo import summary
from utils.dataloader import cifar_10_dataloader,tiny_imagenet_dataloader
from torchvision.transforms import v2
import random
n_classes = 200


def main():

    #train_dataset, test_dataset, train_loader, test_loader = cifar_10_dataloader(64)
    train_dataset, test_dataset, train_loader, test_loader,sampler = tiny_imagenet_dataloader(128)

    mixup = v2.MixUp(num_classes=n_classes)
    cutmix = v2.CutMix(num_classes=n_classes)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("using GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("using Mac mps")
    else:
        device = torch.device("cpu")
        print("using CPU :(")

    #model = resnet18(n_classes=n_classes).to(device)
    model = resnet50(n_classes=n_classes).to(device)

    summary(model, input_data=torch.randn(1,3, 64, 64).to(device))
    
    criterion = CrossEntropyLoss(label_smoothing=0.1)
    

    epochs = 30

    optimizer = SGD(params=model.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr = 0.01,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        epochs=epochs,
        anneal_strategy='cos'
    )

    import os
    os.makedirs("./checkpoints",exist_ok=True)
    if os.path.exists('./checkpoints/final_model.pt'):
        checkpoint = torch.load('./checkpoints/final_model.pt')
        model.load_state_dict(checkpoint['model_params'])
        optimizer.load_state_dict(checkpoint['optimizer_params'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch= 0

    for epoch in range(start_epoch,epochs):
        correct,total = 0, 0
        model.train()
        train_pbar = tqdm(train_loader,desc=f"epoch = {epoch+1}")
        for images, labels in train_pbar:
            if random.random() > 0.5:
                if random.random()>0.5:
                    images, labels = mixup(images, labels)
                else:
                    images, labels = cutmix(images,labels)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            # _, prediction = torch.max(output,dim=1)
            loss= criterion(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # correct+= (prediction==labels).sum().item() # will not work in mix up
            # total+= len(labels)
            train_pbar.set_postfix({
                'train_loss': loss.item(),
                # 'train_accuracy': correct/total,
                'learning_rate':optimizer.param_groups[0]['lr']
            })

        correct, total = 0, 0 
        model.eval()
        test_pbar = tqdm(test_loader,desc=f"epoch = {epoch+1}")
        with torch.no_grad():
            for images, labels in test_pbar:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                _, prediction = torch.max(output,1)
                loss= criterion(output, labels)
                correct+= (prediction==labels).sum().item()
                total+= len(labels)
                test_pbar.set_postfix({
                'test_accuracy': correct/total,
                
            })
        checkpoint = {
                'model_params':model.state_dict(),
                'epoch': epoch+1,
                'optimizer_params':optimizer.state_dict(),
            }
        torch.save(checkpoint,'./checkpoints/final_model.pt')  


if __name__=='__main__':
    main()
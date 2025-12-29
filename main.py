import torchvision
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, AdamW
from tqdm import tqdm
from models.cifar10_resnet_18 import resnet18
from torchinfo import summary
from utils.dataloader import cifar_10_dataloader




def main():

    train_dataset, test_dataset, train_loader, test_loader = cifar_10_dataloader()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("using GPU")
    elif torch.mps.is_available():
        device = torch.device("mps")
        print("using Mac mps")
    else:
        device = torch.device("cpu")
        print("using CPU :(")

    model = resnet18().to(device)

    summary(model, input_data=torch.randn(1,3, 32, 32).to(device))
    
    criterion = CrossEntropyLoss()
    optimizer = AdamW(params=model.parameters(),lr=0.001)
    epochs = 3

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
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            _, prediction = torch.max(output,1)
            loss= criterion(output, labels)
            loss.backward()
            optimizer.step()
            correct+= sum(prediction==labels).item()
            total+= len(labels)
            train_pbar.set_postfix({
                'train_loss': loss.item(),
                'train_accuracy': correct/total
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
                correct+= sum(prediction==labels)
                total+= len(labels)
                test_pbar.set_postfix({
                'test_accuracy': correct/total
            })
        checkpoint = {
                'model_params':model.state_dict(),
                'epoch': epoch+1,
                'optimizer_params':optimizer.state_dict(),
            }
        torch.save(checkpoint,'./checkpoints/final_model.pt')  


if __name__=='__main__':
    main()
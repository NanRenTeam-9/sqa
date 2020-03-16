"""
author:sqa
time:2020/3/16 10:27

"""
import  torch
from torch.utils.data import DataLoader
from torchvision import  datasets
from torchvision import  transforms
from lenet5 import  Lenet5
from torch import  nn



def main():
    batch_size = 16
    cifar_train = datasets.CIFAR10(root='cifar',download=True,train=True,transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]
    ))
    cifar_train = DataLoader(cifar_train,batch_size=batch_size,shuffle=True)
    cifar_test = datasets.CIFAR10(root='cifar',download=True,train=False,transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]))
    cifar_test = DataLoader(cifar_test,batch_size=batch_size,shuffle=True)

    x,label = next(iter(cifar_train))
    print("x:",x.shape,'label:',label.shape)


    device = torch.device('cpu')
    model = Lenet5().to(device)
    criteon = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    print(model)
    for epoch in range(1000):
        model.train()
        for batchidx, (x,y) in enumerate(cifar_train):
            #[b,3,32,32]
            #[b]
            #loss:tensor scalar
            x,label = x.to(device),label.to(device)
            logits = model(x)
            loss = criteon(logits,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(epoch,loss.item())

        model.eval()
        with torch.no_grad():
            #test
            total_correct =0
            total_num =0
            for x,label in cifar_test:
                x,label = x.to(device),label.to(device)
                #[b,10]
                logits = model(x)
                #[b]
                pred = logits.argmax(dim=1)
                #[b] vs[b] =>scalar tensor
                total_correct +=torch.eq(pred,label).float().sum().item()
                total_num += x.size(0)
            acc = total_correct/total_num
            print(epoch, acc)



if __name__ == '__main__':
    main()
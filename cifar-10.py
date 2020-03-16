import  torch
from torch.utils.data import DataLoader
from torchvision import  datasets
from torchvision import  transforms




def main():
    batch_size = 32
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
    cifar_test = DataLoader(cifar_train,batch_size=batch_size,shuffle=True)

    x,label = next(iter(cifar_train))
    print("x:",x.shape,'label:',label.shape)


if __name__ == '__main__':
    main()
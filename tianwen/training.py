import torch 
from torch import nn 
import torchvision 
from torchvision import transforms as T 
from torch import functional as F 

import os
from d2l import torch as d2l
from PIL import Image 
from matplotlib import pyplot as plt 


def Data_Loader(batch_size, data_dir, transform = None):
    """
    Args:
        batch_size (int): Batch size.
        data_dir (str): Data dirction.
        transform (torchvision.transforms): Data transforms. 
    Return:
        train_iter
        test_iter
    """
    if transform is None:
        train_augs = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_augs = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_augs = transform
        test_augs = transform

    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs), 
        batch_size=batch_size, shuffle=True
    )
    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size
    )

    return train_iter, test_iter

def Net_Train(net, lr, num_epochs, train_iter, test_iter):
    """ Net training
    Args:
        net : net work
        lr (float): learning rate
        num_epochs (int): the number of epochs
        train_iter: train dataset 
        test_iter: test dataset
    """
    device = d2l.try_all_gpus()
    print('training on', device)
    net = nn.DataParallel(net, device_ids=device).to(device[0])
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1.2], 
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            features, labels = features.to(device[0]), labels.to(device[0])
            pred = net(features)
            l = loss(pred, labels)
            l.backward()
            optimizer.step()
            timer.stop()

            with torch.no_grad():
                metric.add(l * features.shape[0], d2l.accuracy(pred, labels), features.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i+1) % (num_batches // 5) == 0 or (i+1) == num_batches:
                animator.add(epoch + (i+1) / num_batches, (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')
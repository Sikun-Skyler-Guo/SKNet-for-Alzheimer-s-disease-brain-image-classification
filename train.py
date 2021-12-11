import torch
from torch import nn, optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms.functional import InterpolationMode
from MyDataset import MyDataset
from SKModule import SKNet
import numpy as np
import csv
from BaselineCNNModule import BaselineCNNNet
'''Train model'''

batch_size = 5
epochs = 30
learning_rate = 2e-6
seed = 1
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_to_tensor = transforms.ToTensor()
random_affine_transform = transforms.Compose([
    transforms.RandomAffine(degrees=1, translate=(0.002, 0.002), scale=(0.995, 1.005), shear=None, interpolation=InterpolationMode('bilinear'), fill=0, fillcolor=None, resample=None),
    transforms.ToTensor(),
])
# in train data, healthy 0: unhealthy 1 = 13: 41, to get balanced data, we need to consider the ratio 
train_data = datasets.ImageFolder('./brain_data/train_data', transform=transform_to_tensor)
test_data = datasets.ImageFolder('./brain_data/test_data', transform=transform_to_tensor)

# use a sampler with weight to get balanced data
weights = [41, 13]
balanced_sampler = WeightedRandomSampler(weights=weights, num_samples=378, replacement=True)

#train_loader = DataLoader(train_data, batch_size=16, shuffle=False, sampler=balanced_sampler)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

### make stack
trainLoss = []
trainAcc = []
testLoss = []
testAcc = []

''' two classification'''
net = BaselineCNNNet(2) #SKNet(2)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# or deal with imbalance with weighted loss function
weights = torch.FloatTensor([1/13, 1/41])
criterion = nn.CrossEntropyLoss(weight=weights)
net.to(device)
criterion.to(device)
total_loss = []

for epoch in range(epochs):
    net.train()
    total_loss.clear()
    for batch, (input, label) in enumerate(train_loader):
        input, label = input.to(device), label.to(device)
        logits = net(input)

        loss = criterion(logits, label)
        total_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    correct = 0
    train_loss = 0
    for input, label in train_loader:
        input, label = input.to(device), label.to(device)
        logits = net(input)

        '''crossentropy'''
        train_loss += criterion(logits, label).item() * input.shape[0]
        pred = logits.argmax(dim=1)

        correct += pred.eq(label).float().sum().item()
    train_loss /= len(train_loader.dataset)
    acc = 100. * correct / len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset), acc))
    
    trainLoss.append(train_loss)
    trainAcc.append(acc)
    
    net.eval()
    correct = 0
    test_loss = 0
    for input, label in test_loader:
        input, label = input.to(device), label.to(device)
        logits = net(input)

        '''crossentropy'''
        test_loss += criterion(logits, label).item() * input.shape[0]
        pred = logits.argmax(dim=1)

        correct += pred.eq(label).float().sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    
    testLoss.append(test_loss)
    testAcc.append(acc)

    torch.save(net.state_dict(), "model/SKNet26_{}.pkl".format(epoch))
        
with open('expRecord.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    header = list(range(0, len(trainLoss)))
    writer.writerow(header)
    writer.writerow(trainLoss)
    writer.writerow(trainAcc)
    writer.writerow(testLoss)
    writer.writerow(testAcc)
f.close
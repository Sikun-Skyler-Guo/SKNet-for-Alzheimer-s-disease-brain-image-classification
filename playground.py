
import torch
from torch.utils.data import ConcatDataset
from torchvision import transforms, models, datasets
from torchvision.transforms.functional import InterpolationMode
from matplotlib import pyplot as plt
from PIL import Image

transform_to_tensor = transforms.ToTensor()
# in train data, healthy: unhealthy = 13: 41, to get balanced data, we need to consider the ratio 
original_train_data = datasets.ImageFolder('./brain_data/train_data', transform=transform_to_tensor)
original_test_data = datasets.ImageFolder('./brain_data/test_data', transform=transform_to_tensor)
print(original_train_data.classes)
print(original_train_data.class_to_idx)
#print(original_train_data.imgs[0][1])
#print(original_train_data[-1][1])
"""
healthy_train_data = original_train_data[0]
unhealthy_train_data = original_train_data[-1]
print(unhealthy_train_data)
for i in range(1, len(original_train_data) - 1):
    if original_train_data[i][1] == 0:
        healthy_train_data = ConcatDataset([original_train_data[i], healthy_train_data])
    else:
        unhealthy_train_data = ConcatDataset([original_train_data[i], unhealthy_train_data])

print(unhealthy_train_data[0])
random_affine_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=None, interpolation=InterpolationMode('bilinear'), fill=0, fillcolor=None, resample=None),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
#img = random_affine_transform(unhealthy_train_data[0])
#img = transforms.ToPILImage()(img)
img = Image.open('./brain_data/train_data/healthy/1.png')
img.show()
"""

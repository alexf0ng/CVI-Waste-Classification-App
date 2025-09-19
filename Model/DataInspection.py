import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),             
    transforms.RandomRotation(30),              
    transforms.RandomHorizontalFlip(),          
    transforms.RandomVerticalFlip(p=0.2),       
    transforms.ColorJitter(brightness=0.2, 
                           contrast=0.2, 
                           saturation=0.2),    
    transforms.ToTensor(),                      
    transforms.Normalize([0.5, 0.5, 0.5], 
                         [0.5, 0.5, 0.5])      
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], 
                         [0.5, 0.5, 0.5])
])

data_dir = "dataset" 

dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


images, labels = next(iter(dataloader))
print(f"Batch shape: {images.shape}, Labels: {labels}")
print(f"Classes: {dataset.classes}")
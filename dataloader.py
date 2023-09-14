import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import ConcatDataset
import random

# imagenet dataset

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
imagenet_input_size = 224
    
imagenet_train_transform = transforms.Compose([
    transforms.Resize(imagenet_input_size),
    transforms.CenterCrop(imagenet_input_size),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

test_transform = transforms.Compose([
    transforms.Resize(imagenet_input_size),
    transforms.CenterCrop(imagenet_input_size),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

imagenet_trainset = datasets.ImageNet(
    root='../../torch_datasets/ImageNet_Subset',
    split='train',
    transform=imagenet_train_transform
)

imagenet_testset = datasets.ImageNet(
    root='../../torch_datasets/ImageNet_Subset',
    split='val',
    transform=test_transform
)


# stanford cars dataset

cars_mean = [0.485, 0.456, 0.406]
cars_std = [0.229, 0.224, 0.225]
cars_input_size = 224
    
cars_train_transform = transforms.Compose([
    transforms.Resize(cars_input_size),
    transforms.CenterCrop(cars_input_size),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cars_mean, cars_std)
])


raw_cars_trainset = datasets.StanfordCars(
    root='../../torch_datasets/stanford_car',
    split='train',
    transform=cars_train_transform
)

raw_cars_testset = datasets.StanfordCars(
    root='../../torch_datasets/stanford_car',
    split='test',
    transform=test_transform
)

class cars_dataset():
    def __init__(self, raw_set):
        self.dataset = raw_set

    def __getitem__(self, index):
        img, label = self.dataset[index]
        label += 1000
        return img, label

    def __len__(self):
        return len(self.dataset)

cars_trainset = cars_dataset(raw_cars_trainset)
cars_testset = cars_dataset(raw_cars_testset)

trainset = ConcatDataset([cars_trainset, imagenet_trainset])
testset = ConcatDataset([cars_testset, imagenet_testset])

ideal_trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=128,
    shuffle=True,
    num_workers=2
)

ideal_testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=512,
    shuffle=True,
    num_workers=2
)

cars_trainloader = torch.utils.data.DataLoader(
    raw_cars_trainset,
    batch_size=128,
    shuffle=True,
    num_workers=2
)
cars_testloader = torch.utils.data.DataLoader(
    raw_cars_testset,
    batch_size=128,
    shuffle=True,
    num_workers=2
)

imagenet_trainloader = torch.utils.data.DataLoader(
    imagenet_trainset,
    batch_size=128,
    shuffle=True,
    num_workers=2
)
imagenet_testloader = torch.utils.data.DataLoader(
    imagenet_testset,
    batch_size=128,
    shuffle=True,
    num_workers=2
)

device_idxs = [i for i in range(8144)]
cloud_idxs = [8144 + i for i in range(40000)]
tuning_device_idxs = random.sample(device_idxs, 1000)
tuning_cloud_idxs = random.sample(device_idxs, 5000)
tuning_idxs = tuning_cloud_idxs + tuning_device_idxs

class dataloader():
    def __init__(self):
        self.generate_dataloader()
    
    def generate_dataloader(self):

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=128,
            shuffle=True,
            num_workers=2
        )

        self.cloud_trainloader = torch.utils.data.DataLoader(
            imagenet_trainset,
            # batch_size=128,
            batch_size=100,
            shuffle=True,
            num_workers=2
        )

        self.device_trainloader = torch.utils.data.DataLoader(
            cars_trainset,
            # batch_size=128,
            batch_size=20,
            shuffle=True,
            num_workers=2
        )

        self.tuningloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=128,
            shuffle=False,
            sampler=SubsetRandomSampler(tuning_idxs),
            num_workers=2
        )

        self.testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=256,
            shuffle=True,
            num_workers=2
        )

    def get_dataloader(self, loader_name):
        if loader_name == 'cloud_train':
            return self.cloud_trainloader
        if loader_name == 'device_train':
            return self.device_trainloader
        if loader_name == 'train':
            return self.trainloader
        if loader_name == 'test':
            return self.testloader
        if loader_name == 'finetuning':
            return self.tuningloader

if __name__=='__main__':
    dc_data = dataloader()
    print(len(dc_data.cloud_trainloader))
    print(len(dc_data.testloader))
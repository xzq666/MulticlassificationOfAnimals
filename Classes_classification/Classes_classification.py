import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
from Classes_Network import *
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
import random
from torch import optim
from torch.optim import lr_scheduler
import copy

"""
用于进行模型训练/验证，并调用训练好的模型进行预测
"""

ROOT_DIR = '../Dataset/'
TRAIN_DIR = 'train/'
VAL_DIR = 'val/'
TRAIN_ANNO = 'Classes_train_annotation.csv'
VAL_ANNO = 'Classes_val_annotation.csv'
CLASSES = ['Mammals', 'Birds']


class MyDataset(Dataset):
    """
    自定义数据集
    """
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform
        if not os.path.isfile(self.annotations_file):
            print(self.annotations_file + 'does not exist!')
        self.file_info = pd.read_csv(annotations_file, index_col=0)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.file_info['path'][idx]
        if not os.path.isfile(image_path):
            print(image_path + '  does not exist!')
            return None
        image = Image.open(image_path).convert('RGB')
        label_class = int(self.file_info.iloc[idx]['classes'])
        sample = {'image': image, 'classes': label_class}
        # 使用PyTorch的transforms对图像进行处理
        if self.transform:
            sample['image'] = self.transform(image)
        return sample


# 将多个transform组合起来使用
# 训练的transforms：
# 1、Resize变换图像尺寸 2、RandomHorizontalFlip依概率p水平翻转 3、ToTensor转换为tensor，并除以255归一化至[0-1]
train_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       ])
# 验证的transforms
# 1、Resize变换图像尺寸 2、ToTensor转换为tensor，并除以255归一化至[0-1]
val_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                     transforms.ToTensor()
                                     ])

# 训练集Dataset
train_dataset = MyDataset(root_dir=ROOT_DIR + TRAIN_DIR,
                          annotations_file=TRAIN_ANNO,
                          transform=train_transforms)
# 验证集Dataset
test_dataset = MyDataset(root_dir=ROOT_DIR + VAL_DIR,
                         annotations_file=VAL_ANNO,
                         transform=val_transforms)
# 训练集DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
# 验证集DataLoader
test_loader = DataLoader(dataset=test_dataset)
data_loaders = {'train': train_loader, 'val': test_loader}
# 若有GPU优先使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def visualize_dataset():
    print(len(train_dataset))
    idx = random.randint(0, len(train_dataset))
    sample = train_loader.dataset[idx]
    print(idx, sample['image'].shape, CLASSES[sample['classes']])
    img = sample['image']
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()


# 随机可视化一个图像
visualize_dataset()


def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    """
    训练“纲”分类模型
    :param model: 模型
    :param criterion: 损失函数
    :param optimizer: 优化函数
    :param scheduler: 调整学习率的方法
    :param num_epochs: epochs次数
    :return:
    """
    Loss_list = {'train': [], 'val': []}
    Accuracy_list_classes = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-*' * 10)
        # 每个epoch都分成训练和验证两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            corrects_classes = 0
            for idx, data in enumerate(data_loaders[phase]):
                # print(phase+' processing: {}th batch.'.format(idx))
                inputs = data['image'].to(device)
                labels_classes = data['classes'].to(device)
                optimizer.zero_grad()
                # 对训练集进行autograd追踪，对验证集不进行autograd追踪
                with torch.set_grad_enabled(phase == 'train'):
                    x_classes = model(inputs)
                    # 将x_classes拼接成两列
                    x_classes = x_classes.view(-1, 2)
                    # 返回x_classes中每行的最大值
                    _, preds_classes = torch.max(x_classes, 1)
                    loss = criterion(x_classes, labels_classes)
                    if phase == 'train':
                        loss.backward()
                        # 更新模型
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                corrects_classes += torch.sum(preds_classes == labels_classes)
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)
            epoch_acc_classes = corrects_classes.double() / len(data_loaders[phase].dataset)
            epoch_acc = epoch_acc_classes
            Accuracy_list_classes[phase].append(100 * epoch_acc_classes)
            print('{} Loss: {:.4f}  Acc_classes: {:.2%}'.format(phase, epoch_loss, epoch_acc_classes))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc_classes
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val classes Acc: {:.2%}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    # 保存效果最好的模型
    torch.save(model.state_dict(), 'best_model.pt')
    print('Best val classes Acc: {:.2%}'.format(best_acc))
    return model, Loss_list, Accuracy_list_classes


# 网络模型
network = Net().to(device)
# 优化器
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
# 损失函数
criterion = torch.nn.CrossEntropyLoss()
# 调整学习率方法：每次下降0.1倍
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
model, Loss_list, Accuracy_list_classes = train_model(network, criterion, optimizer, exp_lr_scheduler, num_epochs=100)

x = range(0, 100)
# 生成训练集与验证集上的loss对比图
y1 = Loss_list["val"]
y2 = Loss_list["train"]
plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
plt.legend()
plt.title('train and val loss vs. epoches')
plt.ylabel('loss')
plt.savefig("train and val loss vs epoches.jpg")
plt.close('all')

# 生成训练集与验证集上的acc对比图
y5 = Accuracy_list_classes["train"]
y6 = Accuracy_list_classes["val"]
plt.plot(x, y5, color="r", linestyle="-", marker=".", linewidth=1, label="train")
plt.plot(x, y6, color="b", linestyle="-", marker=".", linewidth=1, label="val")
plt.legend()
plt.title('train and val Classes_acc vs. epoches')
plt.ylabel('Classes_accuracy')
plt.savefig("train and val Classes_acc vs epoches.jpg")
plt.close('all')


# 可视化
def visualize_model(model):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loaders['val']):
            inputs = data['image']
            labels_classes = data['classes'].to(device)
            x_classes = model(inputs.to(device))
            x_classes = x_classes.view(-1, 2)
            _, preds_classes = torch.max(x_classes, 1)
            print(inputs.shape)
            plt.imshow(transforms.ToPILImage()(inputs.squeeze(0)))
            plt.title('predicted classes: {}\n ground-truth classes:{}'.format(CLASSES[preds_classes], CLASSES[labels_classes]))
            plt.show()


visualize_model(model)

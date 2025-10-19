import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 定义数据路径
data_dir = '/media/dell/codes/liyiming/CIL/datasets/scene_slices'

# 定义训练数据和验证数据的转换
data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


# 创建数据集
full_dataset = datasets.ImageFolder(data_dir, data_transforms)

# 划分训练集和验证集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
dataloaders = {'train': train_loader, 'val': val_loader}
# 获取数据及大小
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = full_dataset.classes

# 检查是否使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载预训练的 ResNet50 模型
model = models.resnet50()
pretrained_weights_path = '/media/dell/codes/liyiming/CIL/Experiments/Proposed/ubteacher/classification/resnet50.pth'
model.load_state_dict(torch.load(pretrained_weights_path))


# 修改最后一层全连接层，使其适应三分类
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.33)


# 训练模型
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个epoch有训练和验证两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为验证模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            dataloader = dataloaders[phase]
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播和优化（仅在训练阶段）
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # 统计损失和正确分类数
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    return model


model = train_model(model, criterion, optimizer, scheduler, num_epochs=100)


# 保存模型
torch.save(model.state_dict(), '../engine/pretrained_resnet50.pth')
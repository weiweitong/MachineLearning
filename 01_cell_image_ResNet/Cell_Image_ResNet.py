import pandas as pd
import os
from PIL import Image
from visdom import Visdom
import torch
from torch.utils import data
from torchvision.transforms import transforms
import numpy as np
import torchvision.models as model
import torch.nn as nn
import copy
import pickle

import matplotlib.pyplot as plt


label = {'Homogeneous': [1, 0, 0, 0, 0, 0],
         'Speckled': [0, 1, 0, 0, 0, 0],
         'Nucleolar': [0, 0, 1, 0, 0, 0],
         'Centromere': [0, 0, 0, 1, 0, 0],
         'NuMem': [0, 0, 0, 0, 1, 0],
         'Golgi': [0, 0, 0, 0, 0, 1]}
batch_size = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

img_transformations2 = transforms.Compose([
    transforms.RandomCrop(80, 4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop((80, 80), (0.6, 1), (0.75, 1.25)),
    transforms.ToTensor()
])



class MyDatasets(data.Dataset):
    def __init__(self, image_root, image_name, label, img_files, data_type='train'):
        self.label = label
        self.image_root = image_root
        self.image_name = image_name
        self.data_type = data_type
        self.img_files = img_files
        self.img = []
        for name in image_name:
            img_path = os.path.join(self.image_root, name)
            self.img.append(img_path)
        if self.data_type == 'train':
            self.transform = img_transformations
        elif self.data_type == 'validation':
            self.transform = transforms.ToTensor()
        elif self.data_type == 'test':
            self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        # img = Image.open(self.img[index]).convert('RGB').resize((78, 78), Image.ANTIALIAS)
        img = Image.open(self.img[index]).convert('RGB')
        tag = train_label.loc[int(self.img[index][-9:-4])].item()
        lab = torch.FloatTensor(self.label[tag])
        img = self.transform(img)
        return img, lab

    def __len__(self):
        return len(self.image_name)


# class CellNet(nn.Module):
#     def __init__(self, num_classes=6):
#         super(CellNet, self).__init__()
#
#         # Create 14 layers of the unit with max pooling in between
#         self.unit1 = Unit(in_channels=1, k_size=3, o_channels=78)
#         self.unit2 = Unit(in_channels=78, k_size=3, o_channels=78)
#         self.unit3 = Unit(in_channels=78, k_size=3, o_channels=78)
#
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#
#         self.unit4 = Unit(in_channels=78, k_size=9, o_channels=156)
#         self.unit5 = Unit(in_channels=156, k_size=3, o_channels=156)
#         self.unit6 = Unit(in_channels=156, k_size=3, o_channels=156)
#         self.unit7 = Unit(in_channels=156, k_size=3, o_channels=156)
#
#         self.pool2 = nn.MaxPool2d(kernel_size=3)
#
#         self.unit8 = Unit(in_channels=156, k_size=5, o_channels=512)
#         self.unit9 = Unit(in_channels=512, k_size=3, o_channels=512)
#         self.unit10 = Unit(in_channels=512, k_size=3, o_channels=512)
#         self.unit11 = Unit(in_channels=512, k_size=3, o_channels=512)
#
#         self.pool3 = nn.MaxPool2d(kernel_size=3)
#
#         self.unit12 = Unit(in_channels=512, k_size=3, o_channels=512)
#         self.unit13 = Unit(in_channels=512, k_size=3, o_channels=512)
#         self.unit14 = Unit(in_channels=512, k_size=3, o_channels=512)
#
#         self.avgpool = nn.AvgPool2d(kernel_size=3)
#
#         # Add all the units into the Sequential layer in exact order
#         self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
#                                  , self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
#                                  self.unit12, self.unit13, self.unit14, self.avgpool)
#
#         self.fc1 = nn.Linear(512, 128)
#         self.fc2 = nn.Linear(128, 24)
#         self.fc3 = nn.Linear(24, num_classes)
#
#     def forward(self, input):
#         x = self.net(input)
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.softmax(x, dim=1)


class MyNetwork(nn.Module):
    def __init__(self, model):
        super(MyNetwork, self).__init__()
        self.conv = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(512, 6)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = torch.softmax(x,dim=1)
        return x



# """
# Let's quickly read the CSV and get the annotations in an (N, 2) array
# where N is the number of landmarks.
# """
#
# gt_training = pd.read_csv('./data/gt_training.csv')
#
# # df = pd.DataFrame(columns=["set"])
#
# df_training = pd.DataFrame(columns=["id", "path", "class"])
# df_validation = pd.DataFrame(columns=["id", "path", "class"])
# df_test = pd.DataFrame(columns=["id", "path", "class"])
#
# for n in range(0, len(gt_training)):
# # for n in range(1):
#
#     (img_name, img_class) = gt_training.iloc[n]
#
#     img_name = int(img_name)
#
#     img_id = img_name
#     # print(type(gt_train))
#
#     if img_name < 10:
#         img_name = "0000" + str(img_name) + ".png"
#     elif img_name < 100:
#         img_name = "000" + str(img_name) + ".png"
#     elif img_name < 1000:
#         img_name = "00" + str(img_name) + ".png"
#     elif img_name < 10000:
#         img_name = "0" + str(img_name) + ".png"
#     else:
#         img_name = str(img_name) + ".png"
#
#     img_path1 = "./data/training/" + img_name
#     img_path2 = "./data/validation/" + img_name
#     img_path3 = "./data/test/" + img_name
#     if os.path.exists(img_path1):
#         img_path = img_path1
#         # df.loc[n] = "training"
#         df_training.loc[df_training.shape[0]] = {"id": img_id, "path": img_path, "class": img_class}
#     elif os.path.exists(img_path2):
#         img_path = img_path2
#         # df.loc[n] = "validation"
#         df_validation.loc[df_validation.shape[0]] = {"id": img_id, "path": img_path, "class": img_class}
#     else:
#         img_path = img_path3
#         # df.loc[n] = "test"
#         df_test.loc[df_test.shape[0]] = {"id": img_id, "path": img_path, "class": img_class}
#
#     # print('Image name: {}'.format(img_name))
#     # print('class: {}'.format(img_class))
#     # print("image path: {}".format(img_path))
#
#     # plt.imshow(io.imread(img_path))
#     # plt.show()
# # print(df)
# # print(gt_training)
#
# # gt_training.insert(2, "set", df)
# df_training.to_csv("./data/training.csv")
# df_validation.to_csv("./data/validation.csv")
# df_test.to_csv("./data/test.csv")


def train(mynet, data_train, data_validation, train_criterion, train_optimizer, epochs_count=50):
    train, validation = [], []
    t_corrects, v_corrects = [], []
    schedulr = torch.optim.lr_scheduler.StepLR(train_optimizer, step_size=10, gamma=0.5)

    best_acc = 10.0
    parameter_best_state = copy.deepcopy(mynet.state_dict())
    for epoch in range(epochs_count):
        train_correct = 0.0
        validation_correct = 0.0
        schedulr.step()
        training_loss_value = 0.0
        validation_loss_value = 0.0
        mynet.train()
        for img, lab in data_train:
            img = img.to(device)
            lab = lab.to(device)
            outputs = mynet(img)

            loss = train_criterion(outputs, lab)
            training_loss_value += (loss * len(lab))
            outputs = torch.softmax(outputs, dim=1)
            train_correct += (outputs.argmax(dim=1) == lab.argmax(dim=1)).sum().float()
            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()
            del loss, img, outputs, lab
        mynet.eval()
        with torch.no_grad():
            for img, lab in data_validation:
                img = img.to(device)
                lab = lab.to(device)
                outputs = mynet(img)
                loss = train_criterion(outputs, lab)
                validation_loss_value += (loss * len(lab))
                outputs = torch.softmax(outputs, dim=1)
                validation_correct += (outputs.argmax(dim=1) == lab.argmax(dim=1)).sum().float()
                del loss, img, outputs, lab
        train.append(training_loss_value / 8701)
        validation.append(validation_loss_value / 2175)
        if validation_loss_value.item() / 2175 < best_acc:
            best_acc = validation_loss_value.item() / 2175
            parameter_best_state = copy.deepcopy(mynet.state_dict())

        train_correct = train_correct / 8701.0 * 100.0
        validation_correct = validation_correct / 2175.0 * 100.0
        t_corrects.append(train_correct)
        v_corrects.append(validation_correct)
        print("{:5d} th epoch, training_loss_value: {:.5f}, train_correct: {:.5f}%, "
              "validation_loss_value: {:.5f}, validation_correct:{:.5f}%"
              .format(epoch + 1, training_loss_value / 8701, train_correct,
                      validation_loss_value / 2175, validation_correct))
    mynet.load_state_dict(parameter_best_state)
    return mynet, train, validation, t_corrects, v_corrects


def test(my_net, test_img_set):
    my_net.eval()
    test_loss_value = 0.0
    test_correct_value = 0.0
    for img, lab in test_img_set:
        img = img.to(device)
        lab = lab.to(device)
        outputs = my_net(img)
        loss = criterion(outputs, lab)
        test_loss_value += (loss * len(lab))
        outputs = torch.softmax(outputs, dim=1)
        test_correct_value += (outputs.argmax(dim=1) == lab.argmax(dim=1)).sum().float()
        del img, lab, outputs
    test_loss_value /= len(test_loader.dataset)
    return test_loss_value, test_correct_value


train_label = pd.read_csv('./data/gt_training.csv', index_col=0)
image_train_name = os.listdir('data/training/')
image_validation_name = os.listdir('data/validation/')
image_test_name = os.listdir('data/test/')


train_img_set = MyDatasets('data/training/', image_train_name, label, train_label)
validation_img_set = MyDatasets('data/validation/', image_validation_name, label, train_label, data_type='validation')
test_img_set = MyDatasets('data/test/', image_test_name, label, train_label, data_type='test')
train_loader = data.DataLoader(train_img_set, batch_size=batch_size, shuffle=True)
validation_loader = data.DataLoader(validation_img_set, batch_size=batch_size, shuffle=False)
test_loader = data.DataLoader(test_img_set, batch_size=batch_size, shuffle=False)


resnet18 = model.resnet18(pretrained=True)
net = MyNetwork(resnet18)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)

net = net.to(device)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
epochs = 150

new_net, loss_train, loss_validation, train_corrects, validation_corrects = train(net, train_loader, validation_loader, criterion, optimizer, epochs)


loss_test, correct = test(new_net, test_loader)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    loss_test, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

plt.plot(loss_train, label='loss of train')
plt.plot(loss_validation, label='loss of validation')
plt.legend(loc='upper right')
plt.show()

plt.plot(train_corrects, label='train_corrects')
plt.plot(validation_corrects, label='validation_corrects')
plt.legend(loc='upper right')
plt.show()

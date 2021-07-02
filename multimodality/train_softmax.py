import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets,models
import json
import os
import torch.nn.functional as F
import torch.optim as optim
from model_softmax import vgg
import torch
from nlp import main
from PIL import ImageFile
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
validation_split = .2
shuffle_dataset = True
random_seed= 42

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
path="tensor_task1"
train_inf=np.load(r"E:\workspace\python\zywang_vgg\multimdal\data\{}\train\inf.npz".format(path))
train_inf=torch.from_numpy(train_inf['arr_0']).cuda()
train_not=np.load(r"E:\workspace\python\zywang_vgg\multimdal\data\{}\train\not.npz".format(path))
train_not=torch.from_numpy(train_not['arr_0']).cuda()
val_inf=np.load(r"E:\workspace\python\zywang_vgg\multimdal\data\{}\val\inf.npz".format(path))
val_inf=torch.from_numpy(val_inf['arr_0']).cuda()
val_not=np.load(r"E:\workspace\python\zywang_vgg\multimdal\data\{}\val\not.npz".format(path))
val_not=torch.from_numpy(val_not['arr_0']).cuda()
# print(train_inf['arr_0'],torch.from_numpy(train_inf['arr_0']))
# print(torch.from_numpy(train_inf['arr_0']).shape[0],type(torch.from_numpy(train_inf['arr_0']).shape[0]))
label_train_inf = Variable(torch.zeros(train_inf.shape[0])).cuda()
label_train_not = Variable(torch.ones(train_not.shape[0])).cuda()
label_val_inf = Variable(torch.zeros(val_inf.shape[0])).cuda()
label_val_not = Variable(torch.ones(val_not.shape[0])).cuda()
print(label_val_inf.shape)
print(label_val_not.shape)
traindata=torch.cat([train_inf,train_not],0)
print(traindata.shape)
traindata_label=torch.cat([label_train_inf,label_train_not],0)
# traindata_label=torch.cat([label_train_inf,label_train_not],0)
print(traindata_label.shape)
# print(traindata.shape,traindata_label.shape)
valdata=torch.cat([val_inf,val_not],0)
valdata_label=torch.cat([label_val_inf,label_val_not],0)

batch_size = 32
train_dataset=TensorDataset(traindata,traindata_label)
val_dataset=TensorDataset(valdata,valdata_label)
train_loader=DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0)
val_loader=DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0)


# print(enumerate(train_loader, start=0))

model=vgg(num_classes=2)
pretrained_dict =torch.load('./softmax_train_task1_select_2.pth')
model.load_state_dict(pretrained_dict)
model.to(device)
#
# model_name = "vgg16"
# net = vgg(model_name=model_name, num_classes=2, init_weights=False)
# #missing_keys,unexpected_keys=net.load_state_dict(torch.load('vgg16-397923af.pth'))
# # pretrained_dict =torch.load('./vgg16-397923af.pth')
# pretrained_dict =torch.load('./vgg16Net.pth')
# net_dict = net.state_dict()
# for k in net_dict.keys():
#      print(k + '   ' + str(list(net_dict[k].size())))
# # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict.keys()) and k!='classifier.6.weight'and k!='classifier.6.bias'and k!='classifier.3.weight'and k!='classifier.3.bias'}
# for k in pretrained_dict.keys():
#      print(k + '   ' + str(list(pretrained_dict[k].size())))
#
# net_dict.update(pretrained_dict)
# net.load_state_dict(net_dict)
# #net.fc=nn.Linear(1000,5)
# net.to(device)
#
#
# loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

best_acc = 0.0
save_path = './softmax_train_task2_select_pre.pth'
for epoch in range(1):
    # # train
    # model.train()
    # running_loss = 0.0
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     data= Variable(data)
    #
    #     optimizer.zero_grad()
    #     output = model(data)
    #     # print(type(output))
    #     # print(output)
    #     # print(type(target))
    #     # print(target)
    #     # loss
    #     loss = F.nll_loss(output, target.long())
    #     loss.backward()
    #     # update
    #     optimizer.step()
    #     # print(type(loss.data.item()))
    #     # print(loss.data.item())
    #     # print(len(train_loader))
    #     if batch_idx % 200 == 0:
    #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             epoch, batch_idx * len(data), len(train_loader.dataset),
    #                    100. * batch_idx / len(train_loader), loss.data.item()))
    # # for step, data in enumerate(train_loader, start=0):
    # #     ft, labels = data
    # #     print(len(ft))
    # #     print(labels)
    # #     optimizer.zero_grad()
    # #     outputs = model(ft.to(device))
    # #     loss = loss_function(outputs, labels.to(device))
    # #     loss.backward()
    # #     optimizer.step()
    # #
    # #     # print statistics
    # #     running_loss += loss.item()
    # #     # print train process
    # #     rate = (step + 1) / len(train_loader)
    # #     a = "*" * int(rate * 50)
    # #     b = "." * int((1 - rate) * 50)
    # #     print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    # # print()

    # validate
    model.eval()
    acc = 0.0  # accumulate accurate number / epoch

    with torch.no_grad():
        test_loss = 0
        correct = 0
        # 测试集
        p_a=p_b=n_a=n_b=0.0
        for data, target in val_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target.long()).data.item()
            # get the index of the max
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred=torch.reshape(pred,(1,len(pred)))
            k = Variable(torch.zeros(1, pred.size()[0]).type(torch.LongTensor)).cuda()
            print(pred)
            print(target)
            p_a += ((pred == k[0]) & (target.to(device) == k[0])).sum().item()
            p_b += ((pred == k[0]) & (target.to(device) != k[0])).sum().item()
            n_a += ((pred != k[0]) & (target.to(device) == k[0])).sum().item()
            n_b += ((pred != k[0]) & (target.to(device) != k[0])).sum().item()
            print(acc)
            print(p_a)
            print(p_b)
            print(n_a)
            print(n_b)
        test_loss /= len(val_loader.dataset)
        acc=int(correct)/len(val_loader.dataset)
        print(acc)
        print(p_a)
        print(p_b)
        print(n_a)
        print(n_b)
        # if acc>best_acc:
        #     print(acc)
        #     best_acc=acc
        #     torch.save(model.state_dict(),save_path)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        # for val_data in val_loader:
        #     val_images, val_labels = val_data
        #     optimizer.zero_grad()
        #     outputs = model(val_images.to(device))
        #     predict_y = torch.max(outputs, dim=1)[1]
        #     acc += (predict_y == val_labels.to(device)).sum().item()
        # val_accurate = acc / val_num
        # if val_accurate > best_acc:
        #     best_acc = val_accurate
        #     torch.save(net.state_dict(), save_path)
        # print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
        #       (epoch + 1, running_loss / step, val_accurate))

print(best_acc)
print('Finished Training')

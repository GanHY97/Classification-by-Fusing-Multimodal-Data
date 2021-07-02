import torch.nn as nn
from torchvision import transforms, datasets, models
import json
import os
import torch.optim as optim
from model import vgg
import torch
from torch.autograd import Variable

def string_rename(old_string, new_string, start, end):
    new_string = old_string[:start] + new_string + old_string[end:]
    return new_string




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path

#image_path = data_root + "/cri/"  # flower data set path
image_path =  r"E:/workspace/python/data/test_task2_select/"  # flower data set path
train_dataset = datasets.ImageFolder(root=image_path+"train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
test_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in test_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()

model_name = "vgg16"
net = vgg(model_name=model_name, num_classes=7, init_weights=False)
#missing_keys,unexpected_keys=net.load_state_dict(torch.load('vgg16-397923af.pth'))
#pretrained_dict =torch.load('./vgg16-397923af.pth')
pretrained_dict =torch.load('./vgg16Net_task1_select.pth')
net_dict = net.state_dict()
# for k in net_dict.keys():
#     print(k + '   ' + str(list(net_dict[k].size())))
#
pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict.keys()) and k!='classifier.6.weight'and k!='classifier.6.bias'and k!='classifier.3.weight'and k!='classifier.3.bias'}
#pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict.keys())}
for k in pretrained_dict.keys():
    print(k + '   ' + str(list(pretrained_dict[k].size())))
net_dict.update(pretrained_dict)
net.load_state_dict(net_dict)
#net.fc=nn.Linear(1000,8)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)


best_acc = 0.0
save_path = './{}Net_task2_select_untog.pth'.format(model_name)

for epoch in range(30):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    #p_a=p_b=n_a=n_b=0.0
    with torch.no_grad():
        r=0.0
        p=0.0
        f=0.0
        p_a = p_b = n_a = n_b = 0.0
    #     for i in range(2):
    #         p_a = p_b = n_a = n_b = 0.0
    #         for val_data in validate_loader:
    #             val_images, val_labels = val_data
    #             optimizer.zero_grad()
    #             outputs = net(val_images.to(device))
    #             predict_y = torch.max(outputs, dim=1)[1]
    #
    # #                print(predict_y,predict_y.size()[0])
    #
    #             k = Variable(torch.zeros(7,predict_y.size()[0]).type(torch.LongTensor)).cuda()  # 全0变量
    #             for j in range(2):
    #                 k[j]=k[j].add(j)
    #                 #        print(zes, ' ', ons)
    #                 #        print((predict_y),' ',(val_labels.to(device)))
    #
    # #            print(zes,' ',ons)
    # #            print((predict_y),' ',(val_labels.to(device)))
    #
    #             # print('predict_y',predict_y)
    #             # print("k[i]",k[i])
    #             # print("val_labels.to(device)",val_labels.to(device))
    #
    #             p_a += ((predict_y==k[i])&(val_labels.to(device)==k[i])).sum().item()
    #             p_b += ((predict_y==k[i])&(val_labels.to(device)!=k[i])).sum().item()
    #             n_a += ((predict_y!=k[i])&(val_labels.to(device)==k[i])).sum().item()
    #             n_b += ((predict_y!=k[i])&(val_labels.to(device)!=k[i])).sum().item()
    #             # print(p_a," ",p_b," ",n_a," ",n_b)
    #         p+=p_a/(p_a+p_b)
    #         r+=p_a/(p_a+n_a)
    #         f+=2*p*r/(p+r)
    #     p=p/2.0
    #     r = r / 2.0
    #     f = f / 2.0





#        print(p_a, p_b, n_a, n_b)
        for val_data in validate_loader:
            val_images, val_labels = val_data
            optimizer.zero_grad()
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()

            k = Variable(torch.zeros(1, predict_y.size()[0]).type(torch.LongTensor)).cuda()
            p_a += ((predict_y == k[0]) & (val_labels.to(device) == k[0])).sum().item()
            p_b += ((predict_y == k[0]) & (val_labels.to(device) != k[0])).sum().item()
            n_a += ((predict_y != k[0]) & (val_labels.to(device) == k[0])).sum().item()
            n_b += ((predict_y != k[0]) & (val_labels.to(device) != k[0])).sum().item()

        val_accurate = acc / val_num

        p=(p_a)/(p_a+p_b)
        r=(p_a)/(p_a+n_a)
        f=2*p*r/(p+r)

        print("TP:",p_a,"FP:",p_b,"FN:",n_a,"TN:",n_b)
        if val_accurate > best_acc and val_accurate>0.830:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        # print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f  P: %.3f  R: %.3f  F1: %.3f' %
        #       (epoch + 1, running_loss / step, val_accurate,p,r,f))
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f ' %
          (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')

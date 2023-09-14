import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from tqdm import tqdm
from copy import deepcopy

device = torch.device('cuda')

def run_eval(model, test_loader, local_device=device):
    model = model.to(local_device)
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        model.eval()
        pbar = tqdm(total=len(test_loader))
        for inputs, labels in test_loader:
            inputs = inputs.to(local_device)
            labels = labels.to(local_device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            correct = predicted.eq(labels.view_as(predicted))
            test_acc += correct.sum().item()
            test_loss += loss.item()
            pbar.update(1)
        pbar.close()
    # 统计训练结果
    test_loss /= len(test_loader)
    test_acc /= len(test_loader.dataset)
    return test_loss, 100*test_acc

def run_train(model, train_loader, test_loader, lr=0.01, epochs=10):
    model = nn.DataParallel(model)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # 使用 cuDNN 加速神经网络训练
    cudnn.benchmark = True

    # 训练模型
    for epoch in range(epochs):
        # 开始训练
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        eval_loss, eval_acc = run_eval(model, test_loader)
        print('Epoch: {}, Eval Loss: {:.4f}, Eval Acc: {:.4f}'.format(epoch, eval_loss, eval_acc))
        return model

def train_cloudsubmodel(model, train_loader, test_loader, lr=0.01, epochs=10):
    cloud_submodel = deepcopy(model.cloud_submodel)
    cloud_submodel = nn.DataParallel(cloud_submodel)
    cloud_submodel.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cloud_submodel.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # 使用 cuDNN 加速神经网络训练
    cudnn.benchmark = True

    # 训练模型
    for epoch in range(epochs):
        # 开始训练
        cloud_submodel.train()
        train_loss = 0.0
        train_acc = 0.0
        pbar = tqdm(total=len(train_loader))
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = cloud_submodel(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.update(1)
        pbar.close()
        eval_loss, eval_acc = run_eval(cloud_submodel, test_loader)
        print('Epoch: {}, Eval Loss: {:.4f}, Eval Acc: {:.4f}'.format(epoch, eval_loss, eval_acc))
        model.cloud_submodel = cloud_submodel.module
    return model

def train_controlmodel(model, train_loader, test_loader, lr=0.01, epochs=10):
    cloud_submodel = deepcopy(model.cloud_submodel)
    cloud_submodel = nn.DataParallel(cloud_submodel, device_ids=[0,1,2,3,4,5])
    cloud_submodel.to(device)
    control_model = deepcopy(model.control_model)
    control_model.to("cuda:6")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(control_model.parameters(), lr=lr)

    # 使用 cuDNN 加速神经网络训练
    cudnn.benchmark = True

    # 训练模型
    cloud_submodel.eval()
    for epoch in range(epochs):
        # 开始训练
        control_model.train()
        train_loss = 0.0
        train_acc = 0.0
        pbar = tqdm(total=len(train_loader))
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            cloud_outputs = cloud_submodel(inputs)
            cloud_outputs = cloud_outputs.to("cuda:6")
            inputs = inputs.to("cuda:6")
            control_outputs = control_model(inputs)
            loss = criterion(control_outputs, cloud_outputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.update(1)
        pbar.close()

        train_loss /= len(train_loader)
        eval_loss, eval_acc = run_eval(control_model, test_loader, local_device=torch.device('cuda:6'))
        print('Epoch: {}, Train Loss: {:.4f}, Eval Acc: {:.4f}'.format(epoch, train_loss, eval_acc))

    model.control_model = control_model
    return model

def train_cosubmodel_control(model, train_loader, lr=0.01, epochs=10, freeze_class=False):
    control_model = deepcopy(model.control_model)
    co_submodel = deepcopy(model.co_submodel)
    control_model.to("cuda:0")
    co_submodel.to("cuda:1")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(co_submodel.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(co_submodel.parameters(), lr=lr)

    # 使用 cuDNN 加速神经网络训练
    cudnn.benchmark = True

    # 训练模型
    control_model.eval()
    for epoch in range(epochs):
        # 开始训练
        co_submodel.train()
        train_loss = 0.0
        train_acc = 0.0
        pbar = tqdm(total=len(train_loader))
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to("cuda:0")
            labels = labels.to("cuda:1")
            optimizer.zero_grad()
            control_outputs = control_model(inputs)
            control_outputs = control_outputs.to("cuda:1")
            inputs = inputs.to("cuda:1")
            co_outputs = co_submodel(inputs)
            outputs = control_outputs + co_outputs
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.update(1)
        pbar.close()
    return co_submodel

def finetuning_control(model, train_loader, test_loader, lr=0.01, epochs=10): 
    control_model = deepcopy(model.control_model)
    co_submodel = deepcopy(model.co_submodel)
    control_model.to("cuda:0")
    co_submodel.to("cuda:1")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(co_submodel.classifier[3].parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            
    # 使用 cuDNN 加速神经网络训练
    cudnn.benchmark = True

    # 训练模型
    control_model.eval()
    co_submodel.eval()
    for epoch in range(epochs):
        # 开始训练
        co_submodel.classifier[3].train()
        train_loss = 0.0
        train_acc = 0.0
        pbar = tqdm(total=len(train_loader))
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to("cuda:0")
            labels = labels.to("cuda:1")
            optimizer.zero_grad()
            control_outputs = control_model(inputs)
            control_outputs = control_outputs.to("cuda:1")
            inputs = inputs.to("cuda:1")
            co_outputs = co_submodel(inputs)
            outputs = control_outputs + co_outputs
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.update(1)
        pbar.close()
    return co_submodel
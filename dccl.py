import numpy as np
from copy import deepcopy
from model import serving_model
from dataloader import *
from utils import *
from datetime import datetime
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

class DCCL(object):
    def __init__(self, cloud_epochs=10, kd_epochs=10, distr_rounds=10, cloud_lr=0.01, kd_lr=0.001, distr_cloud_lr=0.01, distr_device_lr=0.01, tuning_epochs=10):        
        # dataloader
        data_loader = dataloader()
        self.cloud_data = data_loader.get_dataloader('cloud_train')
        self.device_data = data_loader.get_dataloader('device_train')
        self.train_data = data_loader.get_dataloader('train')
        self.test_data = data_loader.get_dataloader('test')
        self.tuning_data = data_loader.get_dataloader('finetuning')

        # model
        self.model = serving_model()

        # other settings
        self.cloud_epochs = cloud_epochs
        self.kd_epochs = kd_epochs
        self.distr_rounds = distr_rounds
        self.cloud_lr = cloud_lr
        self.kd_lr = kd_lr
        self.distr_cloud_lr = distr_cloud_lr
        self.distr_device_lr = distr_device_lr
        self.tuning_epochs = tuning_epochs

    def on_cloud_train(self):
        model = deepcopy(self.model)
        model = train_cloudsubmodel(model, self.cloud_data, self.test_data, lr=self.cloud_lr, epochs=self.cloud_epochs)
        return model

    def on_cloud_transfer(self):
        model = deepcopy(self.model)
        model = train_controlmodel(model, self.cloud_data, self.test_data, lr=self.kd_lr, epochs=int(self.kd_epochs/2))
        model = train_controlmodel(model, self.cloud_data, self.test_data, lr=self.kd_lr/10, epochs=int(self.kd_epochs/2))
        return model

    def cloud_cosubmodel_train(self, init_model, epochs=1, lr=0.01, freeze_class=False):
        model = deepcopy(init_model)
        model = train_cosubmodel_control(model, self.cloud_data, lr=lr, epochs=epochs, freeze_class=freeze_class)
        return model

    def device_cosubmodel_train(self, init_model, epochs=1, lr=0.01, freeze_class=False):
        model = deepcopy(init_model)
        model = train_cosubmodel_control(model, self.device_data, lr=lr, epochs=epochs, freeze_class=freeze_class)
        return model

    def fc_finetuning(self, init_model, epochs=1, lr=0.001):
        model = deepcopy(init_model)
        model = finetuning_control(model, self.tuning_data, self.test_data, lr=lr, epochs=epochs)
        return model

    def model_aggregate(self, cloud_model, device_model, w1=0.5, w2=0.5):
        agg_model = deepcopy(cloud_model)

        # aggregate param-list
        param_keys = []
        for key in agg_model.state_dict():
            if "num_batches" not in key and "classifier" not in key: 
                param_keys.append(key)

        # model aggregation
        for key in param_keys:
            agg_model.state_dict()[key] = w1*cloud_model.state_dict()[key]+w2*device_model.state_dict()[key]

        return agg_model

    def model_aggregate_full(self, cloud_model, device_model, w1=0.5, w2=0.5):
        cloud_model.to("cpu")
        device_model.to("cpu")
        agg_model = mobilenet_v3_large()
        agg_model.classifier[3] = nn.Linear(1280, 1196)

        # aggregate param-list
        param_keys = []
        for key in agg_model.state_dict():
            if "num_batches" not in key: 
                param_keys.append(key)

        # model aggregation
        for key in param_keys:
            agg_model.state_dict()[key].copy_(w1*cloud_model.state_dict()[key]+w2*device_model.state_dict()[key])

        return agg_model

    def dccl(self):
        # pre-train with cloud dataloader
        self.model = self.on_cloud_train()
        self.model.save(path='save_models/effv2_serving_model_cloud.pth')

        # knowledge-transfer
        # self.model.load_cloudsubmodel(path='save_models/effv2_cloud_submodel.pth')
        self.model = self.on_cloud_transfer()
        self.model.save(path='save_models/effv2_serving_model_kd.pth')

        # self.model.load(path='save_models/effv2_serving_model_kd.pth')
        # _, test_acc = run_eval(self.model.cloud_submodel, self.test_data)
        # print('cloud training', test_acc)  

        # DCCL        
        # distr learning
        eval_accs = []
        for round in range(self.distr_rounds):
            distr_lr = 0.01
            if round >= 8:
                distr_lr = 0.005

            if round == 0:
                freeze_class = False
            else:
                freeze_class = False

            cloud_dc_cosubmodel = self.cloud_cosubmodel_train(self.model, epochs=1, lr=distr_lr, freeze_class=freeze_class)
            device_dc_cosubmodel = self.device_cosubmodel_train(self.model, epochs=1, lr=distr_lr, freeze_class=freeze_class)
            self.model.co_submodel = self.model_aggregate_full(cloud_dc_cosubmodel, device_dc_cosubmodel, w1=0.5, w2=0.5)
            tuning_cosubmodel = self.fc_finetuning(self.model, epochs=1, lr=distr_lr)
            self.model.co_submodel = tuning_cosubmodel
            _, test_acc = run_eval(self.model, self.test_data)
            print('distr_learning', round+1, test_acc)
            eval_accs.append(test_acc)
        print(eval_accs)
        self.model.save(path='save_models/serving_model_dccl.pth')     
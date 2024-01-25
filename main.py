import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm

from models.den_model import *
from models.vgg_model import *
from models.res_model import *
from models.py_model import *

from prepare_data import *
from args import build_parser

import sys

class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0, dir='./output'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dir = dir

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), '{}/checkpoint_val_{:.2f}.pt'.format(self.dir,val_loss))
        torch.save(model, '{}/loss_best.pt'.format(self.dir))
        self.val_loss_min = val_loss

def lr_scheduler(optimizer, early, l):
    lr = l
    if early.counter > 6:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(train_data, valid_data, test_data, args, repeat_index, aug_rate=0, shuffleFlag=False):

    config = args

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config.mode == "train":
        is_train = True
    else:
        is_train = False
    
    if config.dataset == "cifar100":
        number_of_classes = 100
    else:
        number_of_classes = 10

    post_str = ""
    if aug_rate>0:
        post_str = "-{}".format(aug_rate)
        
    dataset = config.dataset
    
    if config.train_mode != EXP_MODES.DYNAMIC_AUG_ONLY:
        output_dir = "./output/{}/{}/m{}_r{}_aug{}_s-{}".format(dataset,config.train_model, config.train_mode, repeat_index, post_str,shuffleFlag)
    else:
        output_dir = "./output/{}_aug/{}/m{}_r{}_aug".format(dataset,config.train_model, config.train_mode, repeat_index)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    if is_train:                                                                                                                    # 학습
        early = EarlyStopping(patience=config.patience, dir=output_dir)

        logFile= open("{}/log.txt".format(output_dir), "w")

        train_model = config.train_model
        
        if train_model == 'vggnet':                                                                                                 # 학습 모델 준비
            if config.dataset == "mnist" or config.dataset == "fmnist":
                model = VGG("VGG16m", config.dataset, nc=number_of_classes)
            else:
                model = VGG("VGG16", config.dataset, nc=number_of_classes)
        elif train_model == 'resnet':
            model = ResNet50(config.dataset, nc=number_of_classes)
        elif train_model == 'densenet':
            model = DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=number_of_classes, data=config.dataset)
        else:
            model = PyramidNet(dataset=config.dataset, depth=32, alpha=200, num_classes=number_of_classes, bottleneck=True)
        
        model = model.to(device)
        
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, verbose=True)
        loss_arr = []
        
        best_acc = 0
        
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle=True, num_workers=4)
        if config.train_mode == EXP_MODES.DYNAMIC_AUG_ONLY:                                                                                                 # 기존
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=4)
            

        for i in range(config.epochs):                                                                                              # 모델 학습 시작
            model.train()
            print("=====", i, "Step of ", config.epochs, "=====")

            if config.train_mode == EXP_MODES.DYNAMIC_AUG: 
                train_loader = prepare_train_aug_data_per_epoch(config.dataset, train_data, aug_rate, config.batch_size, shuffleFlag, config.train_mode)

            for j, batch in enumerate(train_loader):
                x, y_ = batch[0].to(device), batch[1].to(device)
                #lr_scheduler(optimizer, early)
                optimizer.zero_grad()
                output = model(x)
                loss = loss_func(output,y_)
                loss.backward()
                optimizer.step()

            if i % 10 ==0:
                loss_arr.append(loss.cpu().detach().numpy())

            correct = 0
            total = 0
            valid_loss = 0
            
            model.eval()
            with torch.no_grad():                                                                                                   # 모델 평가
                for image,label in valid_loader:
                    x = image.to(device)
                    y = label.to(device)
                        
                    output = model.forward(x)
                    valid_loss += loss_func(output, y)
                    _,output_index = torch.max(output,1)

                    total += label.size(0)
                    correct += (output_index == y).sum().float()
                logText = "Epoch {:03d}, Valid Acc: {:.2f}%, Valid loss: {:.2f}\n".format(i, 100*correct/total, valid_loss)
                train_acc = "Accuracy against Validation Data: {:.2f}%, Valid_loss: {:.2f}".format(100*correct/total, valid_loss)
                print(logText)
                logFile.write(logText)
                logFile.flush()

                current_acc = (correct / total) * 100
                if current_acc > best_acc:
                    print(" Accuracy increase from {:.2f}% to {:.2f}%. Model saved".format(best_acc, current_acc))
                    best_acc = current_acc
                    torch.save(model, '{}/acc_best.pt'.format(output_dir))
            early(valid_loss, model)

            if early.early_stop:
                print("stop")
                break
            scheduler.step()
        logFile.close()
    else:  
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, shuffle=True, num_workers=4)                                                                                                                        # 모델 추론
        mymodel = '{}/loss_best.pt'.format(output_dir)
        # Test phase
        model = torch.load(mymodel).to(device)
        model.eval()
        correct = 0
        total_cnt = 0
        
        for step, batch in enumerate(test_loader):
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            total_cnt += batch[1].size(0)
            logits = model(batch[0])
            _, predict = logits.max(1)
            correct += predict.eq(batch[1]).sum().item()
            c = (predict == batch[1]).squeeze()

        valid_acc = correct / total_cnt
        print("\nTest Acc : {}, {}".format(valid_acc,output_dir))

        
if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    repeat_num = args.repeat_num

    config = args

    if config.train_mode == EXP_MODES.DYNAMIC_AUG:
        train_data, valid_data, test_data  = loadData(dataset_name=config.dataset, 
                                                        applyDataAug=config.augmentation,
                                                        aug_rate=0.2,
                                                        aug_sh=False)
    else:
        train_data, valid_data, test_data  = load_aug_Data(dataset_name=args.dataset)
        
    aug_pool = [10, 20, 30, 50, 70, 100]
    for index in range(repeat_num):
        if config.train_mode == EXP_MODES.DYNAMIC_AUG:
            for aug_val in aug_pool:
                train(train_data, valid_data, test_data, args=args, repeat_index=index,aug_rate=aug_val,shuffleFlag=True)
                train(train_data, valid_data, test_data, args=args, repeat_index=index,aug_rate=aug_val,shuffleFlag=False)
        else:
            train(train_data, valid_data, test_data, args=args, repeat_index=index,aug_rate=0,shuffleFlag=True)
        #train(train_data, valid_data, test_data, args=args, repeat_index=index, validshuffleFlag=True)

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models.den_model import *
from models.vgg_model import *
from models.res_model import *
from models.py_model import *
from models.shake_resnet import *

from prepare_data import *
from args import build_parser
import csv

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

def train(train_data, valid_data, args, output_dir, aug_data=None, aug_rate=0):
    config = args

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
    if config.dataset == "cifar100":
        number_of_classes = 100
    else:
        number_of_classes = 10 
    
    early = EarlyStopping(patience=config.patience, dir=output_dir)
    logFile= open("{}/log.txt".format(output_dir), "w")
        
    if config.train_model == 'vggnet':                                                                                                 # 학습 모델 준비
            if config.dataset == "mnist" or config.dataset == "fmnist":
                model = VGG("VGG16m", config.dataset, nc=number_of_classes)
            else:
                model = VGG("VGG16", config.dataset, nc=number_of_classes)
    elif config.train_model == 'resnet':
            model = ResNet50(config.dataset, nc=number_of_classes)
    elif config.train_model == 'densenet':
            model = DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=number_of_classes, data=config.dataset)
    elif config.train_model == "shakeshake":
            model = ShakeResNet(24, config.w_base, number_of_classes)
    elif config.train_model == "pyramidnet":
            model = PyramidNet(dataset=config.dataset, depth=32, alpha=200, num_classes=number_of_classes, bottleneck=True)
        
    model = model.to(device)
        
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, verbose=True)
    
    loss_arr = []
    best_acc = 0
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle=True)
            
    my_image = []
    
    for i in range(config.epochs):                                                                                              # 모델 학습 시작
            print("=====", i, "Step of ", config.epochs, "=====")

            if (config.train_mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_1X) or (config.train_mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_2X) or (config.train_mode == EXP_MODES.ORIG_PLUS_VALID_AUG_1X) or (config.train_mode == EXP_MODES.ORIG_PLUS_VALID_AUG_2X):
                train_loader = prepare_train_aug_data_per_epoch(train_data, aug_data, aug_rate, config.batch_size, config.train_mode)
            
            index = 0
            
            model.train()
            
            for batch in train_loader:
                    x, y = batch[0].to(device), batch[1].to(device)
                    optimizer.zero_grad()
                    output = model(x)
                    loss = loss_func(output,y)
                    loss.backward()
                    optimizer.step()
                    if index==0 and config.imageshowflag:
                        my_image.append(batch[0][1])
                    index = index + 1

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

    if config.imageshowflag:
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for i in range(5):
                axes[i].imshow(my_image[i].permute(1, 2, 0))
                axes[i].axis('off')
            plt.show()
            print("end")

def getOutputDir(dataset,train_model,train_mode,repeat_index,transform_index=None,aug_rate=0):

    if aug_rate>0:
        aug_rate_str = "-{}".format(aug_rate)

    if config.train_mode == EXP_MODES.ORIGINAL:
        output_dir = "./output/{}/{}/m{}_r{}_orig".format(dataset,train_model, train_mode, repeat_index)
    elif config.train_mode == EXP_MODES.DYNAMIC_AUG_ONLY:
        output_dir = "./output/{}/{}/m{}_t{}_r{}_aug-only".format(dataset,train_model, train_mode, transform_index, repeat_index)
    elif config.train_mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_1X:
        output_dir = "./output/{}/{}/m{}_t{}_r{}_our_1x{}".format(dataset,train_model, train_mode, transform_index, repeat_index, aug_rate_str)
    elif config.train_mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_2X:
        output_dir = "./output/{}/{}/m{}_t{}_r{}_our_2x{}".format(dataset,train_model, train_mode, transform_index, repeat_index, aug_rate_str)
    elif config.train_mode == EXP_MODES.ORIG_PLUS_VALID_AUG_1X:
        output_dir = "./output/{}/{}/m{}_t{}_r{}_val_1x{}".format(dataset,train_model, train_mode, transform_index, repeat_index, aug_rate_str)
    elif config.train_mode == EXP_MODES.ORIG_PLUS_VALID_AUG_2X:
        output_dir = "./output/{}/{}/m{}_t{}_r{}_val_2x{}".format(dataset,train_model, train_mode, transform_index, repeat_index, aug_rate_str)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("save path : {}".format(output_dir))

    return output_dir


def test(test_data, args, output_dir):
    config = args

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, shuffle=True, num_workers=4)                                                                                                                        # 모델 추론
    mymodel = '{}/loss_best.pt'.format(output_dir)
    # Test phase
    
    model = torch.load(mymodel).to(device)
    model.eval()
    correct = 0
    total_cnt = 0
        
    for batch in test_loader:
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            total_cnt += batch[1].size(0)
            logits = model(batch[0])
            _, predict = logits.max(1)
            correct += predict.eq(batch[1]).sum().item()
            c = (predict == batch[1]).squeeze()

    acc = correct / total_cnt
    print("\nTest Acc : {}, {}".format(acc,output_dir))
    return acc 
    
def save_result(config, result):
    
    my_acc_results, transform_count, aug_pool = result
    
    f = open('./aug_result.csv','a', newline='')
    wr = csv.writer(f)
    if config.train_mode == EXP_MODES.ORIGINAL:
        my_acc_results.insert(0, "original")
        wr.writerow(my_acc_results)
    elif config.train_mode == EXP_MODES.DYNAMIC_AUG_ONLY:
        my_acc_results.insert(0, "aug_only")
        wr.writerow(my_acc_results)
    elif config.train_mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_1X:
        for i in range(transform_count):
            for j, aug_val in enumerate(aug_pool):
                my_acc_results[i][j].insert(0, "aug_1x_t{}_r{}".format(i,aug_val))
                wr.writerow(my_acc_results[i][j])
    elif config.train_mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_2X:
        for i in range(transform_count):
            for j, aug_val in enumerate(aug_pool):
                my_acc_results[i][j].insert(0, "aug_2x_t{}_r{}".format(i,aug_val))
                wr.writerow(my_acc_results[i][j])
    elif config.train_mode == EXP_MODES.ORIG_PLUS_VALID_AUG_1X:
        for i in range(transform_count):
            for j, aug_val in enumerate(aug_pool):
                my_acc_results[i][j].insert(0, "valid_1x_t{}_r{}".format(i,aug_val))
                wr.writerow(my_acc_results[i][j])
    elif config.train_mode == EXP_MODES.ORIG_PLUS_VALID_AUG_2X:
        for i in range(transform_count):
            for j, aug_val in enumerate(aug_pool):
                my_acc_results[i][j].insert(0, "valid_2x_t{}_r{}".format(i,aug_val))
                wr.writerow(my_acc_results[i][j])
    f.close()
    

def test_for_exp(config):
    repeat_num = config.repeat_num

    _, _, test_orig_dataset = load_original_Data(dataset_name=config.dataset)

    my_acc_results = []
    transform_count = len(my_exp_transforms)

    aug_pool = [30,50,70,100]

    if config.train_mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_1X or config.train_mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_2X:
        my_acc_results = [[[] for _ in range(len(aug_pool))] for _ in range(transform_count)]

    if config.train_mode == EXP_MODES.ORIG_PLUS_VALID_AUG_1X or config.train_mode == EXP_MODES.ORIG_PLUS_VALID_AUG_2X:
        my_acc_results = [[[] for _ in range(len(aug_pool))] for _ in range(transform_count)]

    for repeat_index in range(repeat_num):
        if config.train_mode == EXP_MODES.ORIGINAL:
            dir_name = getOutputDir(config.dataset,config.train_model,config.train_mode,repeat_index)
            acc = test(test_orig_dataset, args=config, output_dir=dir_name)
            my_acc_results.append(acc)
        elif config.train_mode == EXP_MODES.DYNAMIC_AUG_ONLY:
            for t_index in range(transform_count):
                dir_name = getOutputDir(config.dataset,config.train_model,config.train_mode,repeat_index,transform_index=t_index)
                acc = test(test_orig_dataset, args=config, output_dir=dir_name)
                my_acc_results[t_index].append(acc)
        elif config.train_mode >= EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_1X and config.train_mode <= EXP_MODES.ORIG_PLUS_VALID_AUG_2X:
            for t_index in range(transform_count):
                for j, aug_val in enumerate(aug_pool):
                    dir_name = getOutputDir(config.dataset,config.train_model,config.train_mode,repeat_index,transform_index=t_index,aug_rate=aug_val)
                    acc = test(test_orig_dataset, args=config, output_dir=dir_name)
                    my_acc_results[t_index][j].append(acc)
                    
    save_result(config, [my_acc_results, transform_count, aug_pool])

def train_for_exp(config):
    valid_rate = 0.2
    repeat_num = config.repeat_num
    
    train_orig_dataset, valid_orig_ataset, _ = load_original_Data(dataset_name=config.dataset, valid_rate=0.2)
    transform_count = len(my_exp_transforms)
    
    aug_rate_pool = [30,50,70,100]
    
    for repeat_index in range(repeat_num):
        if config.train_mode == EXP_MODES.ORIGINAL:
            dir_name = getOutputDir(config.dataset,config.train_model,config.train_mode,repeat_index)
            train(train_orig_dataset, valid_orig_ataset, args=config, output_dir=dir_name)
        elif config.train_mode == EXP_MODES.DYNAMIC_AUG_ONLY:
            for t_index in range(transform_count):
                train_aug_dataset = load_exp_aug_Data(dataset_name=config.dataset, transform_index=t_index, valid_rate=valid_rate)
                dir_name = getOutputDir(config.dataset,config.train_model,config.train_mode,repeat_index,transform_index=t_index)
                train(train_aug_dataset, valid_orig_ataset, args=config, output_dir=dir_name)
        elif config.train_mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_1X or config.train_mode == EXP_MODES.ORIG_PLUS_DYNAMIC_AUG_2X:
            for t_index in range(transform_count):
                train_aug_dataset = load_exp_aug_Data(dataset_name=config.dataset, transform_index=t_index, valid_rate=valid_rate)
                for j, aug_val in enumerate(aug_rate_pool):
                    dir_name = getOutputDir(config.dataset,config.train_model,config.train_mode,repeat_index,transform_index=t_index,aug_rate=aug_val)
                    train(train_orig_dataset, valid_orig_ataset, args=config, output_dir=dir_name, aug_data=train_aug_dataset, aug_rate=aug_val)
                '''
                # Below code is considering for another different kind of implementation
                full_train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=None)
                train_size = int((0.8) * len(full_train_dataset))
                train_dataset = torch.utils.data.Subset(full_train_dataset, range(train_size))
                custom_train_dataset = getCustomDataset(train_dataset, aug_val)
                train(custom_train_dataset, valid_orig_ataset, test_orig_dataset, args=args, repeat_index=index,aug_rate=aug_val,shuffleFlag=True)            
                '''
        elif config.train_mode == EXP_MODES.ORIG_PLUS_VALID_AUG_1X or config.train_mode == EXP_MODES.ORIG_PLUS_VALID_AUG_2X:
            for t_index in range(transform_count):
                train_aug_dataset = load_exp_aug_Data(dataset_name=config.dataset, transform_index=t_index, valid_rate=0)
                for j, aug_val in enumerate(aug_rate_pool):
                    dir_name = getOutputDir(config.dataset,config.train_model,config.train_mode,repeat_index,transform_index=t_index,aug_rate=aug_val)
                    train(train_orig_dataset, valid_orig_ataset, args=config, output_dir=dir_name, aug_data=train_aug_dataset, aug_rate=aug_val)
                '''
                # Below code is considering for another different kind of implementation
                full_train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=None)
                train_size = int((0.8) * len(full_train_dataset))
                train_dataset = torch.utils.data.Subset(full_train_dataset, range(train_size))
                custom_train_dataset = getCustomDataset(train_dataset, aug_val)
                train(custom_train_dataset, valid_orig_ataset, test_orig_dataset, args=args, repeat_index=index,aug_rate=aug_val,shuffleFlag=True)            
                '''

if __name__ == '__main__':
    parser = build_parser()
    config = parser.parse_args()
    if config.mode == "train":
        train_for_exp(config)
    else:
        test_for_exp(config)
import torch

import os
import csv
import shutil
import math
import numpy as np

from aug_transform import *
from exp_mode import EXP_MODES

class CustomCIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, original_transform, augmented_transform, augmentation_rate):
        self.original_dataset = original_dataset
        self.original_transform = original_transform
        self.augmented_transform = augmented_transform
        self.augmentation_rate = augmentation_rate

    def __getitem__(self, index):
        if torch.rand(1).item() < self.augmentation_rate:
            return self.augmented_transform(self.original_dataset[index][0]), self.original_dataset[index][1]
        else:
            return self.original_transform(self.original_dataset[index][0]), self.original_dataset[index][1]

    def __len__(self):
        return len(self.original_dataset)
    
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
    
def getCustomDataset(train_dataset, augmentation_rate,transform_aug):
    custom_train_dataset = CustomCIFAR10Dataset(train_dataset, transform_original, transform_aug, augmentation_rate)
    return custom_train_dataset

def list_files(path):
    files = os.listdir(path)
    return np.asarray(files)

def split_files(oldpath, newpath, classes):
    for name in classes:
        full_dir = os.path.join(os.getcwd(), f"{oldpath}/{name}")

        files = list_files(full_dir)
        total_file = np.size(files,0)
        # We split data set into 3: train, validation and test

        train_size = math.ceil(total_file * 3/4) # 75% for training 

        validation_size = train_size + math.ceil(total_file * 1/8) # 12.5% for validation
        test_size = validation_size + math.ceil(total_file * 1/8) # 12.5x% for testing 

        train = files[0:train_size]
        validation = files[train_size:validation_size]
        test = files[validation_size:]

        move_files(train, full_dir, f"{newpath}/train/{name}")
        move_files(validation, full_dir, f"{newpath}/validation/{name}")
        move_files(test, full_dir, f"{newpath}/test/{name}")

def move_files(files, old_dir, new_dir):
    new_dir = os.path.join(os.getcwd(), new_dir);
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for file in np.nditer(files):
        old_file_path = os.path.join(os.getcwd(), f"{old_dir}/{file}")
        new_file_path = os.path.join(os.getcwd(), f"{new_dir}/{file}")

        shutil.move(old_file_path, new_file_path)
        
def getOutputDir(dataset,train_model,train_mode,repeat_index, w_base, transform_index=None,aug_rate=0, mode=True):

    save_count = 0
    w_base_str = ""
    
    if train_model == "shakeshake":
        w_base_str = "_{}".format(aug_rate, w_base)
    
    while True:
        if mode:
            tmp = "{}_{}".format(train_model, save_count)
            if os.path.exists("./output/{}/".format(dataset) + tmp):
                save_count += 1
            else:
                train_model = tmp
                break
        else:
            tmp = "{}_{}".format(train_model, save_count)
            if os.path.exists("./output/{}/".format(dataset) + tmp):
                save_count += 1
            else:
                tmp = "{}_{}".format(train_model, save_count-1)
                train_model = tmp
                break

    if train_mode == EXP_MODES.ORIGINAL:
        output_dir = "./output/{}/{}/m{}_r{}_orig{}".format(dataset,train_model, train_mode, repeat_index, w_base_str)
    elif train_mode == EXP_MODES.DYNAMIC_AUG_ONLY:
        output_dir = "./output/{}/{}/m{}_t{}_r{}_aug-only{}".format(dataset,train_model, train_mode, transform_index, repeat_index, w_base)
    elif train_mode == EXP_MODES.ORIG_PLUS_VALID_AUG_1X:
        output_dir = "./output/{}/{}/m{}_t{}_r{}_val_1x{}".format(dataset,train_model, train_mode, transform_index, repeat_index, w_base_str)
    elif train_mode == EXP_MODES.ORIG_PLUS_VALID_AUG_2X:
        output_dir = "./output/{}/{}/m{}_t{}_r{}_val_2x{}".format(dataset,train_model, train_mode, transform_index, repeat_index, w_base_str)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("save path : {}".format(output_dir))

    return output_dir

def save_result(config, result):
    
    my_acc_results, transform_count, aug_val = result
    
    f = open('./aug_result.csv','a', newline='')
    wr = csv.writer(f)
    if config.train_mode == EXP_MODES.ORIGINAL:
        my_acc_results.insert(0, "original")
        wr.writerow(my_acc_results)
    elif config.train_mode == EXP_MODES.DYNAMIC_AUG_ONLY:
        my_acc_results.insert(0, "aug_only")
        wr.writerow(my_acc_results)
    elif config.train_mode == EXP_MODES.ORIG_PLUS_VALID_AUG_1X:
        for i in range(transform_count):
            my_acc_results.insert(0, "valid_1x_t{}_r{}".format(i,aug_val))
            wr.writerow(my_acc_results)
    elif config.train_mode == EXP_MODES.ORIG_PLUS_VALID_AUG_2X:
        for i in range(transform_count):
            my_acc_results.insert(0, "valid_2x_t{}_r{}".format(i,aug_val))
            wr.writerow(my_acc_results)
    f.close()

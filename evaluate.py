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

        
def eval(test_data, dataset_name, model_name, train_mode, repeat_index, aug_rate=0, shuffleFlag=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    mode_dict = { 1:'orig', 2:'aug-only', 3:'our_1x', 4:'our_2x', 5:'aug_2x'}
        
    if EXP_MODES.ORIGINAL == train_mode or EXP_MODES.DYNAMIC_AUG_ONLY == train_mode or EXP_MODES.AUG_2X == train_mode:
        output_dir = "./output/{}/{}/m{}_r{}_{}".format(dataset_name,model_name, train_mode, repeat_index, mode_dict[train_mode])
    else:
        output_dir = "./output/{}/{}/m{}_r{}_{}-{}_s-{}".format(dataset_name,model_name, train_mode, repeat_index, mode_dict[train_mode],aug_rate, shuffleFlag)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4)                                                                                                                        # 모델 추론
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
    print("Test Acc : {:.4f}, {}".format(valid_acc,output_dir))
    return valid_acc

        
if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    repeat_num = args.repeat_num
    
    dataset_names = ["cifar10"]
    model_names = ["vggnet"]
    aug_pool = [30, 50, 70, 100]
    
    for dataset_name in dataset_names:
        _, _, test_data, _  = load_original_and_aug_Data(dataset_name=dataset_name)
        for model_name in model_names:
            _, _, test_data, _  = load_original_and_aug_Data(dataset_name=dataset_name)
            if EXP_MODES.ORIGINAL == args.train_mode or EXP_MODES.DYNAMIC_AUG_ONLY == args.train_mode or EXP_MODES.AUG_2X == args.train_mode:
                for repeat_id in range(repeat_num):
                    eval(test_data, dataset_name, model_name, args.train_mode, repeat_index=repeat_id, aug_rate=0, shuffleFlag=True)
            else:
                for aug_val in aug_pool:
                    for repeat_id in range(repeat_num):
                        eval(test_data, dataset_name, model_name, args.train_mode, repeat_index=repeat_id, aug_rate=aug_val, shuffleFlag=True)

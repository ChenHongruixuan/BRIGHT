import sys
sys.path.append('/home/songjian/project/BRIGHT/essd') # change this to the path of your project

import argparse
import os
import time

import numpy as np


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.make_data_loader_semisup import MultimodalDamageAssessmentDatset
from model.MeanTeacher.network import MTNetwork
from datetime import datetime
from model.MeanTeacher.utils.init_func import init_weight, group_weight
from model.MeanTeacher.engine.lr_policy import WarmUpPolyLR

from util_func.metrics import Evaluator
import util_func.lovasz_loss as L
import torch.nn as nn


class Trainer(object):
    """
    Trainer class that encapsulates model, optimizer, and data loading.
    It can train the model and evaluate its performance on a holdout set.
    """

    def __init__(self, args):
        """
        Initialize the Trainer with arguments from the command line or defaults.

        :param args: Argparse namespace containing:
            - dataset, train_dataset_path, holdout_dataset_path, etc.
            - model_type, model_param_path, resume path for checkpoint
            - learning rate, weight decay, etc.
        """
        self.args = args

        # Initialize evaluator for metrics such as accuracy, IoU, etc.
        self.evaluator = Evaluator(num_class=4)


        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
        self.criterion_csst = nn.MSELoss(reduction='mean')
        # Create the deep learning model. Here we show how to use UNet or SiamCRNN.
        self.deep_model = MTNetwork(num_classes=4, criterion=self.criterion,
                    pretrained_model='/home/songjian/project/BRIGHT/essd/pretrained_weight/resnet50-19c8e357.pth',
                    norm_layer=nn.BatchNorm2d)
        init_weight(self.deep_model.branch1.business_layer, nn.init.kaiming_normal_,
                    nn.BatchNorm2d, 1e-5, 0.1,
                    mode='fan_in', nonlinearity='relu')
        init_weight(self.deep_model.branch2.business_layer, nn.init.kaiming_normal_,
                    nn.BatchNorm2d, 1e-5, 0.1,
                    mode='fan_in', nonlinearity='relu')

        self.deep_model.branch1.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.deep_model.branch2.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # group weight and config optimizer
        base_lr = self.args.learning_rate
        
        params_list_l = []
        params_list_l = group_weight(params_list_l, self.deep_model.branch1.backbone, nn.BatchNorm2d, base_lr)
        for module in self.deep_model.branch1.business_layer:
            params_list_l = group_weight(params_list_l, module, nn.BatchNorm2d,base_lr)        # head lr * 10
        self.optimizer_l = optim.AdamW(params_list_l,
                                 lr=base_lr,
                                 weight_decay=args.weight_decay)

        params_list_r = []
        params_list_r = group_weight(params_list_r, self.deep_model.branch2.backbone, nn.BatchNorm2d, base_lr)
        for module in self.deep_model.branch2.business_layer:
            params_list_r = group_weight(params_list_r, module, nn.BatchNorm2d, base_lr)        # head lr * 10
        self.optimizer_r = optim.AdamW(params_list_r,
                                 lr=base_lr,
                                 weight_decay=args.weight_decay)

        # config lr policy
        total_iteration = self.args.max_iters // self.args.train_batch_size
        # self.lr_policy = WarmUpPolyLR(base_lr, 0.9, total_iteration, 1500)
        
        
        self.deep_model = self.deep_model.cuda()

        # Create a directory to save model weights, organized by timestamp.
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join(args.model_param_path, args.dataset, args.model_type + '_' + now_str)

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
   

    def training(self):
        """
        Main training loop that iterates over the training dataset for several steps (max_iters).
        Prints intermediate losses and evaluates on holdout dataset periodically.
        """
        best_mIoU = 0.0
        best_mIoU_fs = 0.0
        best_mIoU_mix = 0.0

        best_round = []
        torch.cuda.empty_cache()
        train_dataset = MultimodalDamageAssessmentDatset(self.args.train_dataset_path, self.args.train_data_name_list, crop_size=self.args.crop_size, max_iters=self.args.max_iters, type='train')
        # train_dataset_few_shot = MultimodalDamageAssessmentDatset(self.args.train_dataset_path, self.args.train_data_name_list_few_shot, crop_size=self.args.crop_size, max_iters=self.args.max_iters // 16, type='train')
        train_dataset_unsup = MultimodalDamageAssessmentDatset(self.args.test_dataset_path, self.args.test_data_name_list, crop_size=self.args.crop_size, max_iters=self.args.max_iters, type='train')

        train_data_loader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=False)
        train_data_loader_unsup = DataLoader(train_dataset_unsup, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=False)
        # train_data_loader_few_shot = DataLoader(train_dataset_few_shot, batch_size=1, shuffle=True, num_workers=1, drop_last=False)

        elem_num = len(train_data_loader)
        train_enumerator = enumerate(train_data_loader)
        train_enumerator_unsup = enumerate(train_data_loader_unsup)
        # train_enumerator_few_shot = enumerate(train_data_loader_few_shot)

        for current_idx in tqdm(range(elem_num)):
            self.optimizer_l.zero_grad()
            self.optimizer_r.zero_grad()

            _, data = train_enumerator.__next__()
            # _, data_few_shot = train_enumerator_few_shot.__next__()
            _, data_unsup = train_enumerator_unsup.__next__()

            pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data
            # pre_change_imgs_few_shot, post_change_imgs_few_shot, _, labels_clf_few_shot, _ = data_few_shot
            pre_change_imgs_unsup, post_change_imgs_unsup, _, _, _ = data_unsup

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            labels_loc = labels_loc.cuda().long()
            labels_clf = labels_clf.cuda().long()
            # pre_change_imgs_few_shot = pre_change_imgs_few_shot.cuda()
            # post_change_imgs_few_shot = post_change_imgs_few_shot.cuda()
            # labels_clf_few_shot = labels_clf_few_shot.cuda().long()

            # pre_change_imgs = torch.cat([pre_change_imgs, pre_change_imgs_few_shot], dim=0)
            # post_change_imgs = torch.cat([post_change_imgs, post_change_imgs_few_shot], dim=0)
            # labels_clf = torch.cat([labels_clf, labels_clf_few_shot], dim=0)

            pre_change_imgs_unsup = pre_change_imgs_unsup.cuda()
            post_change_imgs_unsup = post_change_imgs_unsup.cuda()

            valid_labels_clf = (labels_clf != 255).any()
            if not valid_labels_clf:
               continue
            
            input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1) # if you use UNet
            input_data_unsup = torch.cat([pre_change_imgs_unsup, post_change_imgs_unsup], dim=1) # if you use UNet
            
            s_sup_pred, t_sup_pred = self.deep_model(input_data, step=1, cur_iter=current_idx)
            s_unsup_pred, t_unsup_pred = self.deep_model(input_data_unsup, step=2, cur_iter=current_idx)
            s_pred = torch.cat([s_sup_pred, s_unsup_pred], dim=0)
            t_pred = torch.cat([t_sup_pred, t_unsup_pred], dim=0)

            softmax_pred_s = F.softmax(s_pred, 1)
            softmax_pred_t = F.softmax(t_pred, 1)


            ### Mean Teacher loss ###
            # Perform semi-supervised learning after initial 10000 iterations
            if current_idx < 10000:
                csst_loss = 0
            else:
                csst_loss = self.criterion_csst(softmax_pred_s, softmax_pred_t.detach())
            csst_loss = csst_loss * 100

            ### Supervised loss For Student ###
            loss_sup = self.criterion(s_sup_pred, labels_clf) +  \
                0.75 * L.lovasz_softmax(F.softmax(s_sup_pred, dim=1), labels_clf, ignore=255)  
            
            ### Supervised loss For Teracher. No Backward ###
            loss_sup_t = self.criterion(t_sup_pred, labels_clf) +  \
                0.75 * L.lovasz_softmax(F.softmax(t_sup_pred, dim=1), labels_clf, ignore=255)  

            unlabeled_loss = True

            # lr = self.lr_policy.get_lr(current_idx)

            # self.optimizer_l.param_groups[0]['lr'] = lr
            # self.optimizer_l.param_groups[1]['lr'] = lr
            # for i in range(2, len(self.optimizer_l.param_groups)):
            #     self.optimizer_l.param_groups[i]['lr'] = lr

            loss = loss_sup + csst_loss
            loss.backward()
            self.optimizer_l.step()

            if (current_idx + 1) % 10 == 0:
                print(f'iter is {current_idx + 1}, seg loss (student) is {loss_sup.item()}, seg loss (teacher) is {loss_sup_t.item()}, csst loss is {csst_loss}')
                
                if (current_idx + 1) % 500 == 0:
                    self.deep_model.eval()
                    
                    val_mIoU, val_OA, val_IoU_of_each_class = self.validation()
                    val_mIoU_fs, val_OA_fs, val_IoU_of_each_class_fs = self.validation_few_shot()
                    test_mIoU, test_OA, test_IoU_of_each_class = self.test()

                    if val_mIoU > best_mIoU:
                        torch.save(self.deep_model.state_dict(), os.path.join(self.model_save_path, f'best_model.pth'))
                        best_mIoU = val_mIoU
                        best_round = {
                            'Test event': self.args.test_event_name,
                            'best iter': current_idx + 1,
                            'best mIoU': [val_mIoU * 100, test_mIoU * 100],
                            'best OA': [val_OA * 100, test_OA * 100],
                            'best sub class IoU': (val_IoU_of_each_class * 100, test_IoU_of_each_class * 100)
                        }
                    
                    if val_mIoU_fs > best_mIoU_fs:
                        torch.save(self.deep_model.state_dict(), os.path.join(self.model_save_path, f'best_model_fs.pth'))
                        best_mIoU_fs = val_mIoU_fs
                        best_round_fs = {
                            'Test event': self.args.test_event_name,
                            'best iter': current_idx + 1,
                            'best mIoU': [val_mIoU_fs * 100, test_mIoU * 100],
                            'best OA': [val_OA_fs * 100, test_OA * 100],
                            'best sub class IoU': (val_IoU_of_each_class_fs * 100, test_IoU_of_each_class * 100)
                        }
                    
                    if (val_mIoU_fs + val_mIoU) * 0.5 > best_mIoU_mix:
                        torch.save(self.deep_model.state_dict(), os.path.join(self.model_save_path, f'best_model_mix.pth'))
                        best_mIoU_mix = (val_mIoU_fs + val_mIoU) * 0.5
                        best_round_mix = {
                            'Test event': self.args.test_event_name,
                            'best iter': current_idx + 1,
                            'best mIoU': [val_mIoU * 100, val_mIoU_fs * 100, test_mIoU * 100],
                            'best OA': [val_OA * 100, val_OA_fs * 100, test_OA * 100],
                            'best sub class IoU': (val_IoU_of_each_class * 100, 
                                                   val_IoU_of_each_class_fs * 100, 
                                                   test_IoU_of_each_class * 100)
                        }

                    self.deep_model.train()

        print('The accuracy of the best round selected using Source domain is ', best_round)
        print('The accuracy of the best round selected using Target domain is ', best_round_fs)
        print('The accuracy of the best round selected using Both domain is ', best_round_mix)

    def validation(self):
        print('---------starting validation-----------')
        self.evaluator.reset()
        dataset = MultimodalDamageAssessmentDatset(self.args.val_dataset_path, self.args.val_data_name_list, 1024, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, num_workers=1, drop_last=False)
        torch.cuda.empty_cache()

        with torch.no_grad():
            for _, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()

                input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1) # if you use UNet
                output_clf = self.deep_model(input_data)  # if you use UNet
                # _, output_clf = self.deep_model(pre_change_imgs, post_change_imgs) # If you use SiamCRNN


                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()

                self.evaluator.add_batch(labels_clf, output_clf)

        
        final_OA = self.evaluator.Pixel_Accuracy()
        IoU_of_each_class = self.evaluator.Intersection_over_Union()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        print(f'OA is {100 * final_OA}, mIoU is {100 * mIoU}, sub class IoU is {100 * IoU_of_each_class}')
        return mIoU, final_OA, IoU_of_each_class
    
    
    def validation_few_shot(self):
        print('---------starting validation (Target domain) -----------')
        self.evaluator.reset()
        dataset = MultimodalDamageAssessmentDatset(self.args.val_dataset_path, self.args.val_data_name_list_few_shot, 1024, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, num_workers=1, drop_last=False)
        torch.cuda.empty_cache()

        with torch.no_grad():
            for _, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()

                input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1) # if you use UNet
                output_clf = self.deep_model(input_data)  # if you use UNet
                # _, output_clf = self.deep_model(pre_change_imgs, post_change_imgs) # If you use SiamCRNN


                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()

                self.evaluator.add_batch(labels_clf, output_clf)

        
        final_OA = self.evaluator.Pixel_Accuracy()
        IoU_of_each_class = self.evaluator.Intersection_over_Union()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        print(f'OA is {100 * final_OA}, mIoU is {100 * mIoU}, sub class IoU is {100 * IoU_of_each_class}')
        return mIoU, final_OA, IoU_of_each_class


    def test(self):
        print('---------starting test-----------')
        self.evaluator.reset()
        dataset = MultimodalDamageAssessmentDatset(self.args.test_dataset_path, self.args.test_data_name_list, 1024, None, 'test')
        test_data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, num_workers=1, drop_last=False)
        torch.cuda.empty_cache()

        with torch.no_grad():
            for _, data in enumerate(test_data_loader):
                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()

                input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1) # if you use UNet
                output_clf = self.deep_model(input_data)  # if you use UNet
                # _, output_clf = self.deep_model(pre_change_imgs, post_change_imgs) # If you use SiamCRNN


                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()

                self.evaluator.add_batch(labels_clf, output_clf)
        
        final_OA = self.evaluator.Pixel_Accuracy()
        IoU_of_each_class = self.evaluator.Intersection_over_Union()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        print(f'OA is {100 * final_OA}, mIoU is {100 * mIoU}, sub class IoU is {100 * IoU_of_each_class}')
        return mIoU, final_OA, IoU_of_each_class
    

def get_data_with_prefix(data_list, prefix):
    # 根据前缀过滤影像名
    return [data_name for data_name in data_list if data_name.startswith(prefix)]

def remove_data_with_prefix(data_list, prefix):
    # 从数据列表中删除符合前缀的影像名
    return [data_name for data_name in data_list if not data_name.startswith(prefix)]

def main():
    parser = argparse.ArgumentParser(description="Training on BRIGHT dataset")
    parser.add_argument('--dataset', type=str, default='BRIGHT')

    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--train_data_list_path', type=str)
    parser.add_argument('--val_dataset_path', type=str)
    parser.add_argument('--val_data_list_path', type=str)
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--test_data_list_path', type=str)

    parser.add_argument('--train_data_list_path_few_shot', type=str)
    parser.add_argument('--val_data_list_path_few_shot', type=str)

    parser.add_argument('--test_event_name', type=str)
    
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--crop_size', type=int)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--val_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)

    parser.add_argument('--train_data_name_list_few_shot', type=list)
    parser.add_argument('--val_data_name_list_few_shot', type=list)

    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_param_path', type=str, default='/home/songjian/project/BRIGHT/manuscript/saved_weights')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_workers', type=int)
    args = parser.parse_args()
    # 读取文件
    with open(args.train_data_list_path, "r") as f:
        train_data_name_list = [data_name.strip() for data_name in f]
    # args.train_data_name_list = train_data_name_list
    with open(args.val_data_list_path, "r") as f:
        val_data_name_list = [data_name.strip() for data_name in f]
    # args.val_data_name_list = val_data_name_list
    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f]


    with open(args.train_data_list_path_few_shot, "r") as f:
        train_data_name_list_few_shot = [data_name.strip() for data_name in f]
    with open(args.val_data_list_path_few_shot, "r") as f:
        val_data_name_list_few_shot = [data_name.strip() for data_name in f]
    # args.test_data_name_list = test_data_name_list

    new_test_data = get_data_with_prefix(train_data_name_list, args.test_event_name) + \
                    get_data_with_prefix(val_data_name_list, args.test_event_name) + \
                    get_data_with_prefix(test_data_name_list, args.test_event_name)
    
    new_train_data_name_list_few_shot = get_data_with_prefix(train_data_name_list_few_shot, args.test_event_name)
    new_val_data_name_list_few_shot = get_data_with_prefix(val_data_name_list_few_shot, args.test_event_name)

    new_train_data_name_list = remove_data_with_prefix(train_data_name_list, args.test_event_name)
    new_val_data_name_list = remove_data_with_prefix(val_data_name_list, args.test_event_name)
    new_test_data_name_list = remove_data_with_prefix(test_data_name_list, args.test_event_name)

    

    # 合并新的test数据到train集
    new_train_data = new_train_data_name_list + new_test_data_name_list + new_train_data_name_list_few_shot
    
    args.train_data_name_list = new_train_data
    args.val_data_name_list = new_val_data_name_list
    # 将新的test数据保存为args.test_data_name_list（如果需要）
    args.test_data_name_list = new_test_data
    args.val_data_name_list_few_shot = new_val_data_name_list_few_shot
    args.train_data_name_list_few_shot = new_train_data_name_list_few_shot

    args.test_data_name_list = [item for item in args.test_data_name_list if item not in new_train_data_name_list_few_shot]
    args.test_data_name_list = [item for item in args.test_data_name_list if item not in new_val_data_name_list_few_shot]
    
    print(f'Training {args.model_type} on the BRIGHT under event-level transfer setup (one-shot). Test event is {args.test_event_name}')
    
    print(f'Test data is {args.test_data_name_list}')
    print(f'Train data is {args.train_data_name_list}')
    print(f'Val data (source domain) is {args.val_data_name_list}')
    print(f'Val data (target domain) is {args.val_data_name_list_few_shot}')

    trainer = Trainer(args)
    trainer.training()
if __name__ == '__main__':
    main()
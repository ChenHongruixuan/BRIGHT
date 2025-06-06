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
from dataset.make_data_loader import MultimodalDamageAssessmentDatset
from model.UNet import UNet
from model.SiamCRNN import SiamCRNN
from datetime import datetime

from util_func.metrics import Evaluator
import util_func.lovasz_loss as L


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
        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=4)
        self.evaluator_total = Evaluator(num_class=4)


        # Create the deep learning model. Here we show how to use UNet or SiamCRNN.
        # self.deep_model = UNet(in_channels=6, out_channels=4) 
        self.deep_model = SiamCRNN()

        self.deep_model = self.deep_model.cuda()

        # Create a directory to save model weights, organized by timestamp.
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join(args.model_param_path, args.dataset, args.model_type + '_' + now_str)

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    def training(self):
        """
        Main training loop that iterates over the training dataset for several steps (max_iters).
        Prints intermediate losses and evaluates on holdout dataset periodically.
        """
        best_mIoU = 0.0
        best_round = []
        torch.cuda.empty_cache()
        train_dataset = MultimodalDamageAssessmentDatset(self.args.train_dataset_path, self.args.train_data_name_list, crop_size=self.args.crop_size, max_iters=self.args.max_iters, type='train')
        train_data_loader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=False)
        elem_num = len(train_data_loader)
        train_enumerator = enumerate(train_data_loader)
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            labels_loc = labels_loc.cuda().long()
            labels_clf = labels_clf.cuda().long()

            valid_labels_clf = (labels_clf != 255).any()
            if not valid_labels_clf:
               continue

            # input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1) # if you use UNet
            # output_clf = self.deep_model(input_data)  # if you use UNet
            outout_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs) # If you use SiamCRNN

            self.optim.zero_grad()   

            ce_loss_loc = F.cross_entropy(outout_loc, labels_loc, ignore_index=255) # if you use SiamCRNN
            lovasz_loss_loc = L.lovasz_softmax(F.softmax(outout_loc, dim=1), labels_loc, ignore=255) # if you use SiamCRNN

            ce_loss_clf = F.cross_entropy(output_clf, labels_clf)
            lovasz_loss_clf = L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_clf, ignore=255)      
            
            # final_loss = ce_loss_clf + 0.75 * lovasz_loss_clf # iuf you use UNet
            final_loss = ce_loss_loc + ce_loss_clf + 0.75 * lovasz_loss_clf  + 0.5 * lovasz_loss_loc     # if you use SiamCRNN


            final_loss.backward()

            self.optim.step()

            if (itera + 1) % 10 == 0:
                print(f'iter is {itera + 1}, classification loss is {final_loss.item()}')
                if (itera + 1) % 500 == 0:
                    self.deep_model.eval()
                    loc_f1_score_val, harmonic_mean_f1_val, final_OA_val, mIoU_val, IoU_of_each_class_val = self.validation()
                    loc_f1_score_test, harmonic_mean_f1_test, final_OA_test, mIoU_test, IoU_of_each_class_test = self.test()

                    if mIoU_val > best_mIoU:
                        torch.save(self.deep_model.state_dict(), os.path.join(self.model_save_path, f'best_model.pth'))
                        best_mIoU = mIoU_val
                        best_round = {
                            'best iter': itera + 1,
                            'loc f1 (val)': loc_f1_score_val * 100,
                            'clf f1 (val)': harmonic_mean_f1_val * 100,
                            'OA (val)': final_OA_val * 100,
                            'mIoU (val)': mIoU_val * 100,
                            'sub class IoU (val)': IoU_of_each_class_val * 100,
                            'loc f1 (test)': loc_f1_score_test * 100,
                            'clf f1 (test)': harmonic_mean_f1_test * 100,
                            'OA (test)': final_OA_test * 100,
                            'mIoU (test)': mIoU_test * 100,
                            'sub class IoU (test)': IoU_of_each_class_test * 100
                        }
                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)


    def validation(self):
        print('---------starting validation-----------')
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        self.evaluator_total.reset()
        
        dataset = MultimodalDamageAssessmentDatset(self.args.val_dataset_path, self.args.val_data_name_list, 1024, None, 'test')
        holdout_data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()

        with torch.no_grad():
            for _, data in enumerate(holdout_data_loader):
                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()

                # input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1) # if you use UNet
                # output_clf = self.deep_model(input_data)  # if you use UNet
                output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs) # If you use SiamCRNN

                output_loc = output_loc.data.cpu().numpy()
                output_loc = np.argmax(output_loc, axis=1)
                labels_loc = labels_loc.cpu().numpy()

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()
                
                self.evaluator_loc.add_batch(labels_loc, output_loc)
                output_clf_damage_part = output_clf[labels_loc > 0]
                labels_clf_damage_part = labels_clf[labels_loc > 0]
                self.evaluator_clf.add_batch(labels_clf_damage_part, output_clf_damage_part)
                self.evaluator_total.add_batch(labels_clf, output_clf)
        
        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.evaluator_clf.Damage_F1_score()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        final_OA = self.evaluator_total.Pixel_Accuracy()
        IoU_of_each_class = self.evaluator_total.Intersection_over_Union()
        mIoU = self.evaluator_total.Mean_Intersection_over_Union()
        print(f'OA is {100 * final_OA}, mIoU is {100 * mIoU}, sub class IoU is {100 * IoU_of_each_class}')
        return loc_f1_score, harmonic_mean_f1, final_OA, mIoU, IoU_of_each_class

    def test(self):
        print('---------starting testing-----------')
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        self.evaluator_total.reset()
        
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

                # input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1) # if you use UNet
                # output_clf = self.deep_model(input_data)  # if you use UNet
                output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs) # If you use SiamCRNN

                output_loc = output_loc.data.cpu().numpy()
                output_loc = np.argmax(output_loc, axis=1)
                labels_loc = labels_loc.cpu().numpy()

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()
                
                self.evaluator_loc.add_batch(labels_loc, output_loc)
                output_clf_damage_part = output_clf[labels_loc > 0]
                labels_clf_damage_part = labels_clf[labels_loc > 0]
                self.evaluator_clf.add_batch(labels_clf_damage_part, output_clf_damage_part)
                self.evaluator_total.add_batch(labels_clf, output_clf)
        
        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.evaluator_clf.Damage_F1_score()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        final_OA = self.evaluator_total.Pixel_Accuracy()
        IoU_of_each_class = self.evaluator_total.Intersection_over_Union()
        mIoU = self.evaluator_total.Mean_Intersection_over_Union()
        print(f'OA is {100 * final_OA}, mIoU is {100 * mIoU}, sub class IoU is {100 * IoU_of_each_class}')
        return loc_f1_score, harmonic_mean_f1, final_OA, mIoU, IoU_of_each_class
    

def main():
    parser = argparse.ArgumentParser(description="Training on BRIGHT dataset")

    parser.add_argument('--dataset', type=str, default='BRIGHT')
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--train_data_list_path', type=str)
    parser.add_argument('--val_dataset_path', type=str)
    parser.add_argument('--val_data_list_path', type=str)
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--test_data_list_path', type=str)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=1)

    parser.add_argument('--crop_size', type=int)

    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--val_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)

    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_param_path', type=str, default='/home/songjian/project/BRIGHT/dfc25_benchmark/saved_weights')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_workers', type=int)

    args = parser.parse_args()

    with open(args.train_data_list_path, "r") as f:
        train_data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = train_data_name_list

    with open(args.val_data_list_path, "r") as f:
        val_data_name_list = [data_name.strip() for data_name in f]
    args.val_data_name_list = val_data_name_list
    

    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()

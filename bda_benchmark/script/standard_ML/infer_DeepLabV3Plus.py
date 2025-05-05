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
from model.DeepLabV3Plus import DeepLabV3Plus
from model.SiamCRNN import SiamCRNN
from datetime import datetime

from util_func.metrics import Evaluator
import util_func.lovasz_loss as L


from PIL import Image

import argparse

class Inference:
    def __init__(self, args):
        self.model_path = args.model_path
        self.output_dir = args.output_dir
        # config = get_config(args)
        num_classes = 4
        # Load dataset
        dataset = MultimodalDamageAssessmentDatset(args.test_dataset_path, args.test_data_list, 1024, None, 'test', suffix='.tif')
        self.test_loader = DataLoader(dataset, batch_size=1, num_workers=1, drop_last=False)
        
        # Load model
        self.model = DeepLabV3Plus(in_channels=6, num_classes=num_classes)
        # self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model = self.model.cuda()
        self.model.eval()
        self.color_map = {
            0: (255, 255, 255),       # No damage - black
            1: (70, 181, 121),     # Minor damage - green
            2: (228, 189, 139),   # Major damage - yellow
            3: (182, 70, 69)      # Destroyed - red
        }
        # Overall evaluator
        self.evaluator = Evaluator(num_class=num_classes)
        self.single_evaluator = Evaluator(num_class=num_classes)
        self.evaluator_clf = Evaluator(num_class=num_classes)

        # Disaster-type-specific evaluators
        self.disaster_type_evaluator_dict = {event: Evaluator(num_class=num_classes) for event in self.get_disaster_types()}
        self.disaster_event_evaluator_dict = {event: Evaluator(num_class=num_classes) for event in self.get_disaster_events()}

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if not os.path.exists(os.path.join(self.output_dir, 'original')):
            os.makedirs(os.path.join(self.output_dir, 'original'))
        
        if not os.path.exists(os.path.join(self.output_dir, 'colored')):
            os.makedirs(os.path.join(self.output_dir, 'colored'))

    def get_disaster_events(self):
        """Returns a list of disaster events based on filename prefixes."""
        return [
            "turkey-earthquake", "hawaii-wildfire", "morocco-earthquake",
            "haiti-earthquake", "la_palma-volcano", "congo-volcano",
            "beirut-explosion", "bata-explosion", "libya-flood", 
            "noto-earthquake", "marshall-wildfire", "ukraine-conflict", "myanmar-hurricane", "mexico-hurricane"
        ]
    
    def get_disaster_types(self):
        """Returns a list of disaster events based on filename prefixes."""
        return [
            "earthquake", "wildfire", "volcano", "explosion", "flood", 
            "conflict", "hurricane"
        ]

    def apply_tta_inference(self, model, pre_change_imgs, post_change_imgs):
        """
        Performs test-time augmentations (TTA) on the input images and
        fuses the resulting logits. Returns fused logits for damage classification.
        
        Args:
            model (nn.Module): your model in eval mode
            pre_change_imgs (Tensor): shape [B, C, H, W]
            post_change_imgs (Tensor): shape [B, C, H, W]
        
        Returns:
            Tensor: fused logits with shape [B, num_damage_classes, H, W]
        """
        # Collect logits from each transform
        logits_collection = []
        
        # 1) No transform
        output_clf = model(torch.cat([pre_change_imgs, post_change_imgs], dim=1))  # output_clf is [B, num_damage_classes, H, W]
        logits_collection.append(output_clf)

        # 2) Horizontal flip
        output_clf_hf = model(torch.cat([pre_change_imgs.flip(dims=[3]), post_change_imgs.flip(dims=[3])], dim=1))
        # Unflip the output back
        output_clf_hf = output_clf_hf.flip(dims=[3])
        logits_collection.append(output_clf_hf)

        # 3) Vertical flip
        output_clf_vf = model(torch.cat([pre_change_imgs.flip(dims=[2]), post_change_imgs.flip(dims=[2])], dim=1))
        # Unflip the output
        output_clf_vf = output_clf_vf.flip(dims=[2])
        logits_collection.append(output_clf_vf)

        # 4) 90-degree rotation
        # Note: torch.rot90(img, k, dims=(2,3)) rotates by 90*k degrees
        # dims=(2,3) => H, W
        pre_90 = torch.rot90(pre_change_imgs, 1, dims=(2, 3))
        post_90 = torch.rot90(post_change_imgs, 1, dims=(2, 3))
        output_clf_90 = model(torch.cat([pre_90, post_90], dim=1))
        # invert rotation
        output_clf_90 = torch.rot90(output_clf_90, 3, dims=(2, 3))
        logits_collection.append(output_clf_90)

        # 5) 180-degree rotation
        pre_180 = torch.rot90(pre_change_imgs, 2, dims=(2, 3))
        post_180 = torch.rot90(post_change_imgs, 2, dims=(2, 3))
        output_clf_180 = model(torch.cat([pre_180, post_180], dim=1))
        # invert rotation
        output_clf_180 = torch.rot90(output_clf_180, 2, dims=(2, 3))
        logits_collection.append(output_clf_180)

        # 6) 270-degree rotation
        pre_270 = torch.rot90(pre_change_imgs, 3, dims=(2, 3))
        post_270 = torch.rot90(post_change_imgs, 3, dims=(2, 3))
        output_clf_270 = model(torch.cat([pre_270, post_270], dim=1))
        # invert rotation
        output_clf_270 = torch.rot90(output_clf_270, 1, dims=(2, 3))
        logits_collection.append(output_clf_270)

        # Fuse logits by averaging
        # shape: [B, num_damage_classes, H, W]
        fused_logits = torch.mean(torch.stack(logits_collection, dim=0), dim=0)
        
        return fused_logits

    def run_inference(self):
        print('Starting inference...')
        self.evaluator.reset()
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_loader)):
                self.single_evaluator.reset()
                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, file_name = data
                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                file_name = file_name[0]  # Get the filename as a string
                
                # Predict
                input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
                output_clf = self.model(input_data)

                
                # output_loc = torch.argmax(output_loc, dim=1).squeeze().cpu().numpy()
                # fused_logits = self.apply_tta_inference(self.model, pre_change_imgs, post_change_imgs)
                # fused_logits shape: [B, num_damage_classes, H, W]
                
                # Now get predicted class by argmax
                output = torch.argmax(output_clf, dim=1).squeeze().cpu().numpy().astype(np.uint8)

                
                # output = torch.argmax(output_clf, dim=1).squeeze().cpu().numpy() # * output_loc
                self.save_colored_map(output, file_name)
                self.save_original_map(output, file_name)
                # Add batch for overall evaluation
                labels_clf = labels_clf.squeeze().cpu().numpy()
                labels_loc = labels_loc.squeeze().cpu().numpy()

                # output_clf_damage_part = output[labels_loc > 0]
                # labels_clf_damage_part = labels_clf[labels_loc > 0]
                # self.evaluator_clf.add_batch(labels_clf_damage_part, output_clf_damage_part)

                self.evaluator.add_batch(labels_clf, output)
                self.single_evaluator.add_batch(labels_clf, output)
                print(f'{file_name}: {self.single_evaluator.Mean_Intersection_over_Union()}')
                # Determine disaster event based on filename prefix and add batch to corresponding evaluator
                for disaster_type in self.disaster_type_evaluator_dict.keys():
                    if disaster_type in file_name:
                        self.disaster_type_evaluator_dict[disaster_type].add_batch(labels_clf, output)
                        break  # Only match one event
                
                for event in self.disaster_event_evaluator_dict.keys():
                    if event in file_name:
                        self.disaster_event_evaluator_dict[event].add_batch(labels_clf, output)
                        break  # Only match one event

        # Compute and print overall metrics
        self.compute_and_print_overall_metrics()

        # Compute and print per-disaster event metrics
        self.compute_and_print_disaster_event_metrics()
        self.compute_and_print_disaster_type_metrics()


    def save_original_map(self, prediction, file_name):
        """Saves the colored damage map."""
        # color_map_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        # for cls, color in self.color_map.items():
        #     color_map_img[prediction == cls] = color
        output_path = os.path.join(self.output_dir, 'original', file_name + '_building_damage.png')
        Image.fromarray(prediction).save(output_path)

    def save_colored_map(self, prediction, file_name):
        """Saves the colored damage map."""
        color_map_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        for cls, color in self.color_map.items():
            color_map_img[prediction == cls] = color
        output_path = os.path.join(self.output_dir, 'colored', file_name + '_building_damage.png')
        Image.fromarray(color_map_img).save(output_path)

    def compute_and_print_overall_metrics(self):
        """Computes and prints overall metrics."""
        pixel_accuracy = self.evaluator.Pixel_Accuracy()
        mean_iou = self.evaluator.Mean_Intersection_over_Union()
        
        print("\nOverall Metrics:")
        print(f'Pixel Accuracy: {pixel_accuracy * 100:.2f}%')
        print(f'Mean IoU: {mean_iou * 100:.2f}%')
        print(f'IoU: {self.evaluator.Intersection_over_Union()}')
        # print(f'F1 Score: {len(self.evaluator_clf.Damage_F1_socore()) / np.sum(1.0 / self.evaluator_clf.Damage_F1_socore()) * 100}')

    def compute_and_print_disaster_type_metrics(self):
        """Computes and prints mIoU for each disaster type."""
        print("\nPer-Disaster Type mIoU:")
        average_mIoU = 0
        for disaster_type, evaluator in self.disaster_type_evaluator_dict.items():
            mean_iou = evaluator.Mean_Intersection_over_Union()
            iou_per_class = evaluator.Intersection_over_Union()
            average_mIoU += mean_iou
            print(f"{disaster_type}: mIoU = {mean_iou * 100:.2f}%, IoU = {iou_per_class * 100}")
        print(f"Average mIoU = {average_mIoU / 7 * 100:.2f}%")
    
    def compute_and_print_disaster_event_metrics(self):
        """Computes and prints mIoU for each disaster event."""
        print("\nPer-Event Type mIoU:")
        average_mIoU = 0
        for event, evaluator in self.disaster_event_evaluator_dict.items():
            mean_iou = evaluator.Mean_Intersection_over_Union()
            iou_per_class = evaluator.Intersection_over_Union()
            average_mIoU += mean_iou
            print(f"{event}: mIoU = {mean_iou * 100:.2f}%, IoU = {iou_per_class * 100}")
        print(f"Average mIoU = {average_mIoU / 14 * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on BRIGHT")
    parser.add_argument('--model_path', type=str, default='BRIGHT')
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--test_data_list_path', type=str)
    parser.add_argument('--output_dir', type=str)

    
    args = parser.parse_args()
    
    # Load test data list
    with open(args.test_data_list_path, "r") as f:
        test_data_list = [data_name.strip() for data_name in f]
    args.test_data_list = test_data_list

    inference = Inference(args)
    inference.run_inference()

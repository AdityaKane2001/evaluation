from conversion import *
import cv2

from utils import *

ret = RetinaConverter('./gt/train_annotations.csv')
gt = ret()
#predret = RetinaConverter('result_csv (2).csv')
#predicted = predret()
#pred_craft = CRAFTConverter('result')
#craft_annots = pred_craft()
yolopredret = YOLOConverter('yolo/yolo_test_labels/labels')
yolo_annots = yolopredret()

#pred_mask = MaskTextConverter('./maskts_result/model_finetune_1000_results')
#mask_annots = pred_mask()
#print(mask_annots)
#print(mask_evaluate(gt,mask_annots))
evaluate(gt, yolo_annots)


#print(craft_evaluate(gt, craft_annots,0.1))
#evaluate(gt,predicted,0.5,0)
#TODO Get all MaskTextSpotter, CRAFT annotations
#TODO Note all mAPs for all things
#TODO run script for all of them


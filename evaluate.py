from conversion import *
import cv2

from utils import *

ret = RetinaConverter('valid_annotations (1).csv')
gt = ret()
predret = RetinaConverter('result_csv (2).csv')
predicted = predret()
pred_craft = CRAFTConverter('craft_result')
craft_annots = pred_craft()

print(craft_evaluate(gt, craft_annots,0.1))
#evaluate(gt,predicted,0.5,0)

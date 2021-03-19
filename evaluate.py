from conversion import *
import cv2
from utils import *

ret = RetinaConverter('valid_annotations (1).csv')
gt = ret()
predret = RetinaConverter('result_csv (2).csv')
predicted = predret()
#print(are_same(gt,predicted))

#box_pruning(gt[0])

print(recall(gt,predicted,0.5,0))


#print(predicted)
#print(gt)






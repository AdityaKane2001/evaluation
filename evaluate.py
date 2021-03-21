from conversion import *
import cv2
from utils import *

ret = RetinaConverter('valid_annotations (1).csv')
gt = ret()
predret = RetinaConverter('result_csv (2).csv')
predicted = predret()

print(evaluate(gt,predicted,0.5,0))







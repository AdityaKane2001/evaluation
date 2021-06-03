# evaluation

Repository for evaluation metrics calculation of YOLO, RetinaNet models. Pushed from Pycharm.   

Important functions:

```
# Main evaluation functions
utils.py/get_false_positives() : line 145
utils.py/get_false_negatives() : line 151
utils.py/evaluate() : line 183

# Duplicate box elimination functions
utils.py/box_pruning() : line 129
utils.py/eliminate_low_conf_boxes() : line 115
utils.py/eliminate_overlaping_boxes() : line 99
```

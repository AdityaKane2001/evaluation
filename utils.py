import numpy as np
import cv2
from conversion import *


def are_same(dict1,dict2):
    dict2_keys = dict2.keys()
    for i in dict1.keys():
        if i not in dict2_keys:
            return False
        if not np.array_equal(dict1[i],dict2[i]):
            return False
    return True

# def intersects(box1,box2):
#     x11,x12 = box1[0],box1[2]
#     x21,x22 = box2[0],box2[2]
#     y11, y12 = box1[1], box1[3]
#     y21, y22 = box2[1], box2[3]
#
#     if (x21>=x11 and x21<=x12) or (x22>=x11 and x22<=x12):
#         if (y21>=y11 and y21<=y12) or (y22>=y11 and y22<=y12):
#             return True
#     return False
#
#
# def calc_iou(box1, box2):
#     #[x1,y1,x2,y2]
#     if intersects(box1,box2):
#         all_x = sorted([box1[0],box1[2],box2[0],box2[2]])
#         int_x1,int_x2 = all_x[1],all_x[2]
#         all_y = sorted([box1[1],box1[3],box2[1],box2[3]])
#         int_y1,int_y2 = all_y[1],all_y[2]
#         intersection = abs(int_x1-int_x2)*abs(int_y1-int_y2)
#         union = abs((box1[0]-box1[2])*(box1[1]-box1[3]))+abs((box2[0]-box2[2])*(box2[1]-box2[3])) - intersection
#
#         iou = intersection/union
#         if iou>=0:
#             return iou
#     return 0

def get_overlap_tuples(matrix):
    overlaps=[]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i>j:
                if matrix[i,j]>0 and matrix[i,j]<1:
                    overlaps.append((i,j))

    return overlaps

def eliminate_overlaping_boxes(overlap_tuples, annots_boxes):
    final_boxes = [list(i) for i in annots_boxes]
    annot_boxes = [list(i) for i in annots_boxes]

    for j in range(len(overlap_tuples)):
        tup = overlap_tuples[j]
        if annot_boxes[tup[0]][4] > annot_boxes[tup[1]][4]:
            if annot_boxes[tup[1]] in final_boxes:
                final_boxes.remove(annot_boxes[tup[1]])
        else:
            if annot_boxes[tup[0]] in final_boxes:
                final_boxes.remove(annot_boxes[tup[0]])


    return final_boxes

def eliminate_low_conf_boxes(boxes,conf_thresh = 0.25):
    final_boxes = [i for i in boxes]


    if len(final_boxes)!= 0:

        for j in range(len(final_boxes)):

            if final_boxes[j][4] < conf_thresh:
                 final_boxes.pop(j)

            return final_boxes
    return boxes

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA ) * max(0, yB - yA )
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] ) * (boxA[3] - boxA[1] )
    boxBArea = (boxB[2] - boxB[0] ) * (boxB[3] - boxB[1] )
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def compute_overlap(a,b):
    overlap_matrix = np.zeros((a.shape[0],b.shape[0]))
    for i in range(len(a)):
        for j in range(len(b)):
            overlap_matrix[i,j] = iou(a[i],b[j])

    return overlap_matrix


def box_pruning(boxes,conf_thresh=0.5):

    if boxes.shape!=(0,):

        overlap = compute_overlap(boxes,boxes)

        pruned_boxes = eliminate_overlaping_boxes(get_overlap_tuples(overlap), boxes)

        pruned_boxes = eliminate_low_conf_boxes(pruned_boxes,conf_thresh=conf_thresh)

        return np.array(pruned_boxes)
    return boxes

def draw_caption(image, box, caption, mode='up'):
    if mode == 'up':
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    if mode == 'down':
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[3] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[3] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def draw_predictions(image_name,gt_boxes,pred_boxes):
    img = cv2.imread(os.path.join('valid',image_name))
    img[img < 0] = 0
    img[img > 255] = 255
    if len(gt_boxes)!=0:
        for i in range(len(gt_boxes)):
            bbox = gt_boxes[i]
            try:
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = 'Questions'
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            except:
                print(bbox)

    if len(pred_boxes)!=0:
        for k in range(len(pred_boxes)):
            bbox = pred_boxes[k]
            try:


                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])


                draw_caption(img, (x1, y1, x2, y2), 'GT', mode='down')

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            except:
                print(bbox)

    cv2.imwrite(os.path.join('predictions', image_name + '.jpg'), img)

def recall(gt,predicted,iou_threshold=0.5,conf_thresh=0.5):
    false_negatives = 0
    false_positives = 0
    true_positives = 0
    num_annotations = 0

    num_gt_annots = 0
    num_pred_annots = 0

    for i in predicted.keys():
        gt_key = [key for key in gt.keys() if i[:25] in key]
        #print(i)
        if gt_key!=[]:


            gt_annots = gt[gt_key[0]]

            pred_annots = predicted[i]
            pred_annots = box_pruning(pred_annots,conf_thresh=conf_thresh)

            draw_predictions(i,gt_annots,pred_annots)

            num_gt_annots += len(gt_annots)
            num_pred_annots += len(pred_annots)

            gt_annots = np.array(sorted(gt_annots, key=lambda x: x[1]))
            pred_annots = np.array(sorted(pred_annots, key=lambda x: x[1]))


            #gt_annots = non_max_suppression_fast(gt_annots,0.95)
            #print('GT annots for this image:',len(gt_annots))
            #print('Pred annots for this image before NMS:',len(pred_annots))
            #pred_annots = nms(pred_annots,0.5)
            # TODO : conf threshold
            # TODO : use most conf box
            #print('Pred annots for this image after NMS:', len(pred_annots))
            #print(pred_annots.shape)
            #random.shuffle(pred_annots)
            overlaps = compute_overlap(gt_annots,pred_annots) #(gt_len,pred_len)
            #print(overlaps)
            #if overlaps.shape[0]!=overlaps.shape[1]:
            #    print('GT key: ',gt_key)
            #    print('predicted key: ',i)
            #print(overlaps.shape)
            #print('Overlaps: ',overlaps)
            # print('Get overlaps: ',get_overlap(gt_annots,pred_annots))


            if pred_annots.shape==(0,):

                if len(gt_annots)==0:
                    num_annotations+=1
                    true_positives+=1



                if len(gt_annots) > 0:
                    num_annotations += 1
                    false_negatives += 1

            else:
                num_annotations += len(gt_annots)
                true_positives += len(np.where(overlaps>iou_threshold)[0])



        else:
            pass

    print('num_gt_annots: ',num_gt_annots)
    print('num_pred_annots: ',num_pred_annots)
    print('true_positives: ',true_positives)
    print('num_annotations: ',num_annotations)
    return true_positives/num_annotations



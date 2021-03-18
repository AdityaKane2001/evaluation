from conversion import *
import random

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

def eliminate_boxes(overlap_tuples, annots_boxes):
    final_boxes = [list(i) for i in annots_boxes]
    annot_boxes = [list(i) for i in annots_boxes]
    print(overlap_tuples)
    print(final_boxes)
    print(annot_boxes)
    for j in range(len(overlap_tuples)):
        tup = overlap_tuples[j]
        if annot_boxes[tup[0]][4] > annot_boxes[tup[1]][4]:
            if annot_boxes[tup[1]] in final_boxes:
                final_boxes.remove(annot_boxes[tup[1]])
        else:
            if annot_boxes[tup[0]] in final_boxes:
                final_boxes.remove(annot_boxes[tup[0]])


    return final_boxes

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

def nms(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]

def compute_overlap(a,b):
    overlap_matrix = np.zeros((a.shape[0],b.shape[0]))
    for i in range(len(a)):
        for j in range(len(b)):
            overlap_matrix[i,j] = iou(a[i],b[j])

    return overlap_matrix


def box_pruning(boxes):
    pruned_boxes = []

    overlap = compute_overlap(boxes,boxes)

    print(eliminate_boxes(get_overlap_tuples(overlap), boxes))



    return np.array(pruned_boxes)




def recall(gt,predicted,iou_threshold):
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
            print(i)
            pred_annots = predicted[i]
            box_pruning(pred_annots)
            return 0

            num_gt_annots += len(gt_annots)
            num_pred_annots += len(pred_annots)

            gt_annots = np.array(sorted(gt_annots, key=lambda x: x[1]))
            pred_annots = np.array(sorted(pred_annots, key=lambda x: x[1]))

            #gt_annots = non_max_suppression_fast(gt_annots,0.95)
            #print('GT annots for this image:',len(gt_annots))
            #print('Pred annots for this image before NMS:',len(pred_annots))
            pred_annots = nms(pred_annots,0.5)
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


            if overlaps.all()==None:

                if len(gt_annots)==0 and len(pred_annots)==0:
                    num_annotations+=1
                    true_positives+=1

                if len(gt_annots) == 0 and len(pred_annots) > 0:
                    num_annotations += 1
                    false_positives += 1

                if len(gt_annots) > 0 and len(pred_annots) == 0:
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



ret = RetinaConverter('valid_annotations (1).csv')
gt = ret()
predret = RetinaConverter('result_csv (2).csv')
predicted = predret()
#print(are_same(gt,predicted))

#box_pruning(gt[0])

print(recall(gt,predicted,0.5))

#print(predicted)
#print(gt)






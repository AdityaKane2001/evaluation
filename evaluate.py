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

def intersects(box1,box2):
    x11,x12 = box1[0],box1[2]
    x21,x22 = box2[0],box2[2]
    y11, y12 = box1[1], box1[3]
    y21, y22 = box2[1], box2[3]

    if (x21>=x11 and x21<=x12) or (x22>=x11 and x22<=x12):
        if (y21>=y11 and y21<=y12) or (y22>=y11 and y22<=y12):
            return True
    return False


def calc_iou(box1, box2):
    #[x1,y1,x2,y2]
    if intersects(box1,box2):
        all_x = sorted([box1[0],box1[2],box2[0],box2[2]])
        int_x1,int_x2 = all_x[1],all_x[2]
        all_y = sorted([box1[1],box1[3],box2[1],box2[3]])
        int_y1,int_y2 = all_y[1],all_y[2]
        intersection = abs(int_x1-int_x2)*abs(int_y1-int_y2)
        union = abs((box1[0]-box1[2])*(box1[1]-box1[3]))+abs((box2[0]-box2[2])*(box2[1]-box2[3])) - intersection

        recall = intersection/union
        if recall>=0:
            return recall
    return 0

def compute_overlap(a,b):
    overlap_matrix = np.zeros((a.shape[0],b.shape[0]))
    for i in range(len(a)):
        for j in range(len(b)):
            overlap_matrix[i,j] = calc_iou(a[i],b[j])

    return overlap_matrix



'''
def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua
'''


def recall(gt,predicted,iou_threshold):
    false_negatives = 0
    false_positives = 0
    true_positives = 0
    num_annotations = 0

    for i in predicted.keys():
        gt_key = [key for key in gt.keys() if i[:20] in key][0]
        gt_annots = gt[gt_key]
        pred_annots = predicted[i]

        gt_annots = np.array(sorted(gt_annots, key=lambda x: x[1]))
        pred_annots = np.array(sorted(pred_annots, key=lambda x: x[1]))

        #print(pred_annots.shape)
        #random.shuffle(pred_annots)
        overlaps = compute_overlap(gt_annots,pred_annots) #(gt_len,pred_len)
        if overlaps.shape[0]!=overlaps.shape[1]:
            print('GT key: ',gt_key)
            print('predicted key: ',i)
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
            #print(overlaps)
            true_positives += len(np.where( overlaps>=iou_threshold))
            #TODO find false positives

    return true_positives/true_positives + false_negatives



ret = RetinaConverter('valid_annotations.csv')
gt = ret()
predicted = ret()
#print(are_same(gt,predicted))

print(recall(gt,predicted,0.5))
#print(predicted)
#print(gt)






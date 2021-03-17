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

        iou = intersection/union
        if iou>=0:
            return iou
    return 0

def compute_overlap(a,b):
    overlap_matrix = np.zeros((a.shape[0],b.shape[0]))
    for i in range(len(a)):
        for j in range(len(b)):
            overlap_matrix[i,j] = calc_iou(a[i],b[j])

    return overlap_matrix


def get_true_positives(matrix):
    tp = 0
    #print(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j]>0.4:
                tp+=1
    return tp


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
            pred_annots = predicted[i]

            num_gt_annots += len(gt_annots)
            num_pred_annots += len(pred_annots)

            gt_annots = np.array(sorted(gt_annots, key=lambda x: x[1]))
            pred_annots = np.array(sorted(pred_annots, key=lambda x: x[1]))

            #print(pred_annots.shape)
            #random.shuffle(pred_annots)
            overlaps = compute_overlap(gt_annots,pred_annots) #(gt_len,pred_len)
            #print(overlaps)
            #if overlaps.shape[0]!=overlaps.shape[1]:
            #    print('GT key: ',gt_key)
            #    print('predicted key: ',i)
            #print(overlaps.shape)

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

                true_positives += get_true_positives(overlaps)
        else:
            pass

    print('num_gt_annots: ',num_gt_annots)
    print('num_pred_annots: ',num_pred_annots)
    print('true_positives: ',true_positives)
    print('num_annotations: ',num_annotations)
    return true_positives/num_annotations



ret = RetinaConverter('valid_annotations.csv')
gt = ret()
predret = RetinaConverter('result_csv.csv')
predicted = predret()
#print(are_same(gt,predicted))

print(recall(gt,predicted,0.5))

#print(predicted)
#print(gt)






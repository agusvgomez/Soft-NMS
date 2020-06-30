import numpy as np

"""
This is a Python version used to implement the Soft NMS algorithm.
Original Paper：Improving Object Detection With One Line of Code
"""

def get_iou(box1, box2):
    """ return intersection over union result between two boxes (format[x1,y1,x2,y2])
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    area1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1) 
    area2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)
    xA = max(x1_1, x1_2)
    yA = max(y1_1, y1_2)
    xB = min(x2_1, x2_2)
    yB = min(y2_1, y2_2)
    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    union = float(area1 + area2 - intersection)
    return intersection / union
    
def soft_nms(boxes, scores, overlap_treshold=0.3, sigma=0.5, score_treshold=0.001, method='linear'):
    """
    soft_nms
    :param boxes:   boxes format [x1,y1,x2,y2]
    :param sc:      boxe scores
    :param overlap_treshold:  iou treshold
    :param sigma:  for gaussian implementation
    :param thresh:  score treshold
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    if boxes.dtype.kind == "float":
        boxes = boxes.astype("int")

    N = boxes.shape[0]
    if N==0:
        return []
    
    for i in range(N):
        sorted_idxs = (-scores[i:]).argsort()
        sorted_boxes_c = boxes[i:][sorted_idxs]
        
        iou = [get_iou(sorted_boxes_c[0], box) for box in sorted_boxes_c[1:]]

        if method=='nms':
            weights = [1 if iou_score<overlap_treshold else 0 for iou_score in iou]
        elif method=='linear':
            weights = [1 if iou_score<overlap_treshold else (1-iou_score) for iou_score in iou]
        else:
            weights = [np.exp(-(iou_score * iou_score) / sigma) for iou_score in iou]

        scores[i+1:] = scores[i+1:] * weights

    selected = boxes[scores > score_treshold].astype(int)
    return selected


if __name__=="__main__":
    boxes = np.array([[200, 200, 400, 400], [220, 220, 420, 420], [240, 200, 440, 400], [200, 240, 400, 440], [1, 1, 2, 2]], dtype=np.int)
    scores = np.array([0.9, 0.8, 0.7, 0.85, 0.5], dtype=np.float32)
    print('boxes to filter: ',boxes)
    print()
    print('selected boxes',soft_nms(boxes, scores, method='nms'))
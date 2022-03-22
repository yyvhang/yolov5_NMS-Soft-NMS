import torch
import math
import pdb
import copy
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
 
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
 
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
 
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
 
    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def Self_NMS(boxes, scores, iou_thres, GIoU=False, DIoU=False, CIoU=False):
    """
    :param boxes:  (Tensor[N, 4])): are expected to be in ``(x1, y1, x2, y2)
    :param scores: (Tensor[N]): scores for each one of the boxes
    :param iou_thres: discards all overlapping boxes with IoU > iou_threshold
    :return:keep (Tensor): int64 tensor with the indices
            of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    # 按conf从大到小排序
    B = torch.argsort(scores, dim=-1, descending=True) ##返回的是索引值
    keep = []
    while B.numel() > 0:
        # 取出置信度最高的
        index = B[0]
        keep.append(index)
        if B.numel() == 1: break
        # 计算iou,根据需求可选择GIOU,DIOU,CIOU
        iou = bbox_iou(boxes[index, :], boxes[B[1:], :], GIoU=GIoU, DIoU=DIoU, CIoU=CIoU)
        # 找到符合阈值的下标
        inds = torch.nonzero(iou <= iou_thres).reshape(-1)
        ##这里主要是处理一个索引的问题，这里计算iou的时候是用score最高的box与其他box计算，得到的iou是个列表
        ##但此时的iou列表已经比B少了一个值了，ins返回的是iou < 阈值的框在iou列表里面的索引
        ##这时要返回其在B中的索引就需要把inds+1，然后得到进行一轮NMS后剩下的框
        B = B[inds + 1]
    return torch.tensor(keep)

def Self_Soft_NMS(boxes, scores, iou_thres, Giou=False, DIou=False, CIou=False, weight_method=0, sigma=0.5, soft_threshold=0.25):
    turn_scores = copy.deepcopy(scores)
    B = torch.argsort(turn_scores, dim=-1, descending=True)
    keep = []
    while B.numel() > 0:
        B = torch.argsort(turn_scores,dim=-1,descending=True)
        keep_len = len(keep)
        max_index = B[keep_len]
        keep.append(max_index)
        other_index = B[keep_len+1:]
        if(len(other_index)==1):
            keep.append(other_index[-1])
            break
        elif(len(other_index)==0):
            break
        iou = bbox_iou(boxes[max_index, :], boxes[B[other_index], :],GIoU=Giou, DIoU=DIou,CIoU=CIou)
        inds = torch.nonzero(iou > iou_thres).reshape(-1)
        #线性
        if(weight_method == 0):
            turn_scores[B[inds+(keep_len+1)]] *= (1-iou[inds])
        #高斯
        elif(weight_method == 1):
            turn_scores[B[inds+(keep_len+1)]] *= torch.exp(-(torch.pow(iou[inds],2) / sigma))

    final_inds = []
    for idx in keep:
        if(turn_scores[idx] > soft_threshold):
            final_inds.append(idx)


    return torch.tensor(final_inds)
    
boxes = torch.tensor([[31089.98633, 30926.14844, 31121.43945, 30960.42188],
        [31090.14062, 30926.00391, 31121.24219, 30959.99023],
        [31089.76172, 30926.07422, 31121.28320, 30960.60547],
        [46447.39844, 46092.57422, 46490.67188, 46117.88672],
        [46447.46875, 46092.78125, 46490.35156, 46117.74609],
        [46447.48047, 46092.98828, 46490.17188, 46117.84766],
        [46449.80078, 46110.52734, 46493.84766, 46140.70312],
        [46449.66406, 46110.51172, 46493.92578, 46140.73438],
        [46449.82812, 46110.25781, 46493.88672, 46140.24219],
        [31082.91406, 30775.50977, 31133.59180, 30815.77344],
        [31083.01367, 30776.77344, 31134.39844, 30816.02148],
        [31083.03711, 30776.49219, 31134.09766, 30816.25391],
        [31083.04883, 30776.59570, 31134.49609, 30816.50586],
        [46445.67969, 46173.13281, 46500.12891, 46209.06641],
        [46445.51562, 46173.19141, 46500.25391, 46208.66797]])

scores = torch.tensor([0.96707, 0.97318, 0.96200, 0.96735, 0.95249, 0.96769, 0.94777, 0.95450, 0.89021, 0.26787, 0.96995, 0.94509, 0.90604, 0.94727, 0.97445])

i = Self_Soft_NMS(boxes, scores, 0,45)
print(i)

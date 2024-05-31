from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import skimage.io as io
import matplotlib.pyplot as plt
import os
import copy

# 初始化COCO ground truth数据
cocoGt = COCO('../mydev/home/wjy/MDMTcross/annotations/test_cocoformat.json')  # 替换为你的标注json文件路径

# 初始化COCO预测结果
cocoDt = cocoGt.loadRes('work_dirs/coco_detection/test.bbox.json')  # 替换为你的预测结果json文件路径

# 初始化COCO评估对象
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

# 对所有类别进行迭代
catIds = cocoGt.getCatIds()
print('catIds', catIds)
all_class_aps = {}
for catId in catIds:
    # 创建一个新的COCOeval对象，并设置只评估当前类别
    cocoEval_specific = copy.deepcopy(cocoEval)
    cocoEval_specific.params.catIds = [catId]
    
    # 运行评估
    cocoEval_specific.evaluate()
    cocoEval_specific.accumulate()
    cocoEval_specific.summarize()
    
    # 存储当前类别的AP
    class_name = cocoGt.loadCats(catId)[0]['name']
    all_class_aps[class_name] = cocoEval_specific.stats[0]

# 输出所有类别的mAP
for class_name, ap in all_class_aps.items():
    print(f"mAP for class {class_name}: {ap}")

# 计算并输出所有类别的平均mAP
mean_ap = sum(all_class_aps.values()) / len(all_class_aps)
print(f"Mean mAP across all classes: {mean_ap}")
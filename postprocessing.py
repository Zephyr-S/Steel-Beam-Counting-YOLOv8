from typing import Tuple, Dict
import cv2
import numpy as np
from PIL import Image
from ultralytics.yolo.utils.plotting import colors
import random
from typing import Tuple
from ultralytics.yolo.utils import ops
import torch
import numpy as np

def plot_one_box_xz(box:np.ndarray, img:np.ndarray, color:Tuple[int, int, int] = None, mask:np.ndarray = None, label:str = None, line_thickness:int = 5):
    """
    Helper function for drawing single bounding box on image
    Parameters:
        x (np.ndarray): bounding box coordinates in format [x1, y1, x2, y2]
        img (no.ndarray): input image
        color (Tuple[int, int, int], *optional*, None): color in BGR format for drawing box, if not specified will be selected randomly
        mask (np.ndarray, *optional*, None): instance segmentation mask polygon in format [N, 2], where N - number of points in contour, if not provided, only box will be drawn
        label (str, *optonal*, None): box label string, if not provided will not be provided as drowing result
        line_thickness (int, *optional*, 5): thickness for box drawing lines
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    if mask is not None:
        image_with_mask = img.copy()
        mask
        cv2.fillPoly(image_with_mask, pts=[mask.astype(int)], color=color)
        img = cv2.addWeighted(img, 0.5, image_with_mask, 0.5, 1)
    return img

def print_report_xz(boxes:np.ndarray, img:np.ndarray, line_thickness:int = 1):
    print('Adding report on the image...')
    count_sum = {'0': 0, '1': 0, '2': 0}
    h, w = img.shape[:2]
    print('the h and w of img is: ', h, w)
    cv2.rectangle(img, (0, h-130), (150, h), (105, 105, 105), -1)  # cv2.rectangle(img, (0, w-130), (150, w), (105, 105, 105), -1)
    for idx, (*xyxy, conf, lbl) in enumerate(boxes):
        if lbl == 0:  # I-beam
            count_sum['0']+=1
        if lbl == 1:  # O-beam
            count_sum['1']+=1
        if lbl == 2:  # T-beam
            count_sum['2']+=1
    print(count_sum)
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)  # font thickness
    # report = 'I-beam: %s \nI-beam: %s \nI-beam: %s\n' % (count_sum['0'], count_sum['1'], count_sum['2'])
    cv2.putText(img, 'I-beam:'+str(count_sum['0']), (5, h-90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), line_thickness)
    cv2.putText(img, 'O-beam:'+str(count_sum['1']), (5, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), line_thickness)
    cv2.putText(img, 'T-beam:'+str(count_sum['2']), (5, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), line_thickness)
    return img, count_sum

def draw_results_xz(results:Dict, source_image:np.ndarray, label_map:Dict):
    """
    Helper function for drawing bounding boxes on image
    Parameters:
        image_res (np.ndarray): detection predictions in format [x1, y1, x2, y2, score, label_id]
        source_image (np.ndarray): input image for drawing
        label_map; (Dict[int, str]): label_id to class name mapping
    Returns:

    """
    boxes = results["det"]  # 模型预测的检测框
    masks = results.get("segment")  # 模型预测的mask
    h, w = source_image.shape[:2]  # 获得原始图片的长和宽
    for idx, (*xyxy, conf, lbl) in enumerate(boxes):  # 对每个检测框
        # print('here', xyxy, conf, lbl)  # 坐标；置信度；labelid （0就是I-beam）
        label = f'{label_map[int(lbl)][0]}-{idx+1} {conf:.2f}'  # 打印的x-beam和置信度；label_map: {0: 'I-beam', 1: 'O-beam', 2: 'T-beam'}
        mask = masks[idx] if masks is not None else None
        source_image = plot_one_box_xz(xyxy, source_image, mask=mask, label=label, color=colors(int(lbl)), line_thickness=1)
    source_image, count_sum = print_report_xz(boxes, source_image)
    return source_image, count_sum

def add_target(img:np.ndarray, add_point_1:list, add_point_2:list, add_class:int, count_sum, color:Tuple[int, int, int] = None, line_thickness:int = 1):
  # 1.根据输入坐标和种类画方框和label
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    cv2.rectangle(img, add_point_1, add_point_2, color, thickness=tl, lineType=cv2.LINE_AA)
    tf = max(tl - 1, 1)  # font thickness
    label_map = {0: 'I-beam', 1: 'O-beam', 2: 'T-beam'}
    add_class_str = label_map[add_class]
    t_size = cv2.getTextSize(add_class_str, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = add_point_1[0] + t_size[0], add_point_1[1] - t_size[1] - 3
    cv2.rectangle(img, add_point_1, c2, color, -1, cv2.LINE_AA)  # filled

    cv2.putText(img, add_class_str, (add_point_1[0], add_point_1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
  # 2.重新打印report
    h, w = img.shape[:2]
    print('the h and w of img is: ', h, w)
    cv2.rectangle(img, (0, h-130), (150, h), (105, 105, 105), -1)  # cv2.rectangle(img, (0, w-130), (150, w), (105, 105, 105), -1)
    if add_class == 0:  # I-beam
        count_sum['0']+=1
    if add_class == 1:  # O-beam
        count_sum['1']+=1
    if add_class == 2:  # T-beam
        count_sum['2']+=1
    print(count_sum)
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)  # font thickness
    # report = 'I-beam: %s \nI-beam: %s \nI-beam: %s\n' % (count_sum['0'], count_sum['1'], count_sum['2'])
    cv2.putText(img, 'I-beam:'+str(count_sum['0']), (5, h-90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), line_thickness)
    cv2.putText(img, 'O-beam:'+str(count_sum['1']), (5, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), line_thickness)
    cv2.putText(img, 'T-beam:'+str(count_sum['2']), (5, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), line_thickness)
    return img, count_sum


try:
    scale_segments = ops.scale_segments
except AttributeError:
    scale_segments = ops.scale_coords

def postprocess(
    pred_boxes:np.ndarray,
    input_hw:Tuple[int, int],
    orig_img:np.ndarray,
    min_conf_threshold:float = 0.25,
    nms_iou_threshold:float = 0.7,
    agnosting_nms:bool = False,
    max_detections:int = 300,
    pred_masks:np.ndarray = None,
    retina_mask:bool = False
):
    """
    YOLOv8 model postprocessing function. Applied non maximum supression algorithm to detections and rescale boxes to original image size
    Parameters:
        pred_boxes (np.ndarray): model output prediction boxes
        input_hw (np.ndarray): preprocessed image
        orig_image (np.ndarray): image before preprocessing
        min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
        nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
        agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
        max_detections (int, *optional*, 300):  maximum detections after NMS
        pred_masks (np.ndarray, *optional*, None): model ooutput prediction masks, if not provided only boxes will be postprocessed
        retina_mask (bool, *optional*, False): retina mask postprocessing instead of native decoding
    Returns:
       pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label] and segment - segmentation polygons for each element in batch
    """
    nms_kwargs = {"agnostic": agnosting_nms, "max_det":max_detections}
    # if pred_masks is not None:
    #     nms_kwargs["nm"] = 32
    # print(torch.from_numpy(pred_boxes).shape[1], min_conf_threshold, nms_iou_threshold)
    preds = ops.non_max_suppression(  # error
        torch.from_numpy(pred_boxes),
        min_conf_threshold,
        nms_iou_threshold,
        nc=3,  ## 3 ## 80     必须要根据自己的数据集情况修改!!!!!!!!!!!!!!!
        **nms_kwargs
    )
    results = []
    proto = torch.from_numpy(pred_masks) if pred_masks is not None else None

    for i, pred in enumerate(preds):
        shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
        if not len(pred):
            results.append({"det": [], "segment": []})
            continue
        if proto is None:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            results.append({"det": pred})
            continue
        if retina_mask:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], shape[:2])  # HWC
            segments = [scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], input_hw, upsample=True)
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            segments = [scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        results.append({"det": pred[:, :6].numpy(), "segment": segments})
    return results

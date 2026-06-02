import time
from typing import List, Optional, Union

import numpy as np
import supervision as sv  # 要求 python >= 3.8


def draw_mask(
    img_src: np.ndarray, masks: np.ndarray, mask_opacity: float = 0.5
) -> np.ndarray:
    """绘制分割掩码到图像上"""
    detections = sv.Detections(mask=masks.astype(bool))
    annotator = sv.MaskAnnotator(opacity=mask_opacity, color=sv.ColorPalette.DEFAULT)
    return annotator.annotate(img_src.copy(), detections)


def draw_result(
    img_src: np.ndarray,
    boxes: Optional[Union[np.ndarray, List[List[float]], List[float]]] = None,
    class_names: Optional[Union[List[str], str]] = None,
    scores: Optional[Union[np.ndarray, List[float], float]] = None,
    masks: Optional[np.ndarray] = None,
    box_thickness: int = 2,
    text_scale: float = 0.5,
    text_thickness: int = 1,
) -> np.ndarray:
    """
    智能绘制检测结果，根据输入参数自动判断绘制哪些元素

    Args:
        img_src: 输入图像 (H, W, 3)
        boxes: 可选，边界框，支持以下格式:
            - numpy数组 (n, 4) in [x1, y1, x2, y2]
            - 单个框的列表 [x1, y1, x2, y2]
            - 多个框的列表 [[x1, y1, x2, y2], ...]
        class_names: 可选，类别名称，必须与检测结果数量一致:
            - 列表 [cls1, cls2, ...] (长度为n)
            - 单个字符串 "cls1" (当n=1时)
        scores: 可选，置信度分数，必须与检测结果数量一致:
            - numpy数组 (n,)
            - 列表 [score1, score2, ...] (长度为n)
            - 单个浮点数 score1 (当n=1时)
        masks: 可选，分割掩码 (n, H, W)
        其他参数同前...

    Returns:
        标注后的图像 (H, W, 3)

    Raises:
        ValueError: 当boxes/masks与class_names/scores数量不匹配时
    """
    start_time = time.time()
    # 参数校验
    if boxes is None and masks is None:
        raise ValueError("至少需要提供 boxes 或 masks 之一")

    # 处理boxes输入格式并获取检测数量
    if boxes is not None:
        if isinstance(boxes, list):
            if len(boxes) == 0:  # 空列表
                boxes = np.zeros((0, 4))
                num_detections = 0
            elif isinstance(boxes[0], (int, float)):  # 单个框 [x1,y1,x2,y2]
                boxes = np.array([boxes])
                num_detections = 1
            else:  # 多个框 [[x1,y1,x2,y2], ...]
                boxes = np.array(boxes)
                num_detections = len(boxes)
        else:  # numpy数组
            boxes = boxes.reshape(-1, 4) if boxes.size > 0 else np.zeros((0, 4))
            num_detections = len(boxes)
    else:
        boxes = np.zeros((0, 4))
        num_detections = 0

    # 处理masks输入格式并获取检测数量
    if masks is not None:
        masks = np.asarray(masks)
        num_detections = len(masks)
        if boxes is not None and len(boxes) > 0 and len(boxes) != num_detections:
            raise ValueError("boxes和masks的数量不一致")
    else:
        masks = None

    # 检查检测数量是否一致
    if num_detections == 0 and (class_names is not None or scores is not None):
        raise ValueError("检测数量为0但提供了class_names或scores")

    # 处理class_names输入格式并校验数量
    if class_names is not None:
        if isinstance(class_names, str):  # 单个类别名
            class_names = [class_names]
        else:  # 列表或其他可迭代对象
            class_names = list(class_names)
            if len(class_names) != num_detections:
                raise ValueError(
                    f"class_names数量({len(class_names)})与检测数量({num_detections})不匹配"
                )
    else:
        class_names = None

    # 处理scores输入格式并校验数量
    if scores is not None:
        if isinstance(scores, (int, float)):  # 单个分数
            scores = np.array([scores])
        else:  # 列表或numpy数组
            scores = np.asarray(scores).reshape(-1)
            if len(scores) != num_detections:
                raise ValueError(
                    f"scores数量({len(scores)})与检测数量({num_detections})不匹配"
                )
    else:
        scores = None

    # 准备标签文本
    labels = None
    if class_names is not None or scores is not None:
        labels = []
        for i in range(num_detections):
            parts = []
            if class_names is not None:
                parts.append(str(class_names[i]))
            if scores is not None:
                parts.append(f"{scores[i]:.2f}")
            labels.append(" ".join(parts))

    # 创建Detections对象
    detections = sv.Detections(
        xyxy=boxes,
        mask=masks.astype(bool) if masks is not None else None,
        class_id=np.arange(num_detections) if num_detections > 0 else np.array([]),
        confidence=scores if scores is not None else None,
    )

    # 初始化标注器
    annotators = []
    if masks is not None:
        annotators.append(sv.MaskAnnotator())

    if boxes is not None and num_detections > 0:
        annotators.append(sv.BoxAnnotator(thickness=box_thickness))

        if labels is not None:
            annotators.append(
                sv.LabelAnnotator(text_scale=text_scale, text_thickness=text_thickness)
            )

    # 按顺序执行标注
    annotated_img = img_src.copy()
    for annotator in annotators:
        if isinstance(annotator, sv.LabelAnnotator):
            annotated_img = annotator.annotate(annotated_img, detections, labels=labels)
        else:
            annotated_img = annotator.annotate(annotated_img, detections)
    end_time = time.time()
    cost_time = end_time - start_time
    return annotated_img, cost_time

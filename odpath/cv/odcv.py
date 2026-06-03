import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


# 展示图片
def show(img, win_name="image", time=0):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    if time >= 0:
        cv2.waitKey(time)


# 把点值int化
def int_pt(pt):
    int_pt = (int(pt[0]), int(pt[1]))
    return int_pt


def base64_to_numpy(data):
    """
    功能：将base64编码的字符串转换为numpy数组
    参数：
        data (str): base64编码的字符串
    返回：
        numpy.ndarray: 转换后的numpy数组
    """
    img_bytes = base64.b64decode(data)
    nparr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_data = np.ascontiguousarray(img_data)
    return img_data

def base64_to_pil(base64_str: str) -> Image.Image:
    """
    将 Base64 字符串转换为 PIL Image 对象
    :param base64_str: Base64 编码的字符串（支持带 Data URI 前缀）
    :return: PIL 图片对象
    """
    # 1. 如果包含 Data URI 前缀 (如 data:image/jpeg;base64,)，利用正则去掉它
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
    # 2. 将字符串解码为字节流
    img_bytes = base64.b64decode(base64_str)
    # 3. 将字节流转为 BytesIO 对象并用 PIL 打开
    image = Image.open(BytesIO(img_bytes))
    # 建议加上 .convert("RGB")，防止某些格式（如 RGBA 或 P 模式）在后续处理中报错
    return image.convert("RGB")



def pil_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    将 PIL Image 对象转换为 Base64 字符串
    :param image: PIL 图片对象
    :param format: 存储格式 (JPEG, PNG, WEBP 等)
    :return: Base64 编码的字符串
    """
    # 1. 创建一个字节流容器
    buffered = BytesIO()
    # 2. 将 PIL 图片保存到字节流中，如果是 JPEG 格式，建议设置 quality
    image.save(buffered, format=format)
    # 3. 获取字节流内容并进行 Base64 编码
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def numpy_to_base64_pil(data):
    """
    功能：将numpy数组转换为base64编码的字符串
    参数：
        data (numpy.ndarray): 要转换的numpy数组
    返回：
        str: base64编码的字符串
    """
    # 确保数据类型为uint8
    if data.dtype != np.uint8:
        data = data.astype(np.uint8)
    pil_img = Image.fromarray(data)
    output_buffer = BytesIO()
    pil_img.save(output_buffer, format="jpeg")
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str


def numpy_to_base64(data, format="jpg"):
    if data.dtype != np.uint8:
        data = data.astype(np.uint8)

    if format == "jpg":
        # JPEG 格式：控制质量 (Lossy), JPEG 压缩会损失细节，但文件极小。
        # [cv2.IMWRITE_JPEG_QUALITY, 质量级别], 从 0 到 100，默认是 95。数值越低，文件越小，画质越差。
        success, buffer = cv2.imencode(".jpg", data, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif format == "png":
        # PNG 格式：控制压缩率 (Lossless), PNG 是无损的，但你可以控制压缩算法的耗时。
        # [cv2.IMWRITE_PNG_COMPRESSION, 压缩级别], 从 0 到 9，默认是 3。数值越高，压缩时间越长，但文件越小。
        success, buffer = cv2.imencode(
            ".png", data, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        )
    if not success:
        return None
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return base64_str


# Byte 转 Numpy
def byte_to_numpy(data):
    """
    将字节流转换为numpy数组。
    参数:
        data : 文件对象（例如通过 open(path, 'rb') 打开的文件，或者是 Flask/Django 接收到的上传文件对象,它调用了 .read() 方法将流中的内容提取成字节。也可以是纯 bytes 类型字节流数据。
    返回:
        numpy.ndarray: 转换后的numpy数组。
    """
    # 如果输入是文件对象，先读取内容
    if hasattr(data, "read"):
        data = data.read()
    # 1. 将 bytes 转换为一维 uint8 数组
    nparr = np.frombuffer(data, np.uint8)
    # 2. 解码成图像矩阵
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("解码失败，请检查数据是否为有效的图片格式")
    return data


def numpy_to_byte(data):
    """
    将numpy数组转换为字节流。
    参数:
        data (numpy.ndarray): 要转换的numpy数组。
    返回:
        bytes: 转换后的字节流。
    """
    # 确保数据类型为uint8
    data = data.astype(np.uint8)

    pil_img = Image.fromarray(data)
    output_buffer = BytesIO()
    pil_img.save(output_buffer, format="jpeg")
    byte_data = output_buffer.getvalue()
    return byte_data


def letterbox(
    img,
    new_shape=(640, 640),
    color=(0, 0, 0),
    auto=False,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)

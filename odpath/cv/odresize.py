import cv2
import numpy as np


class ODResize:
    """
    轻量级图像处理工具类，支持多种填充模式（如边缘复制）。
    """

    def __init__(self, color=(114, 114, 114)):
        self.color = color
        # 映射常用的填充模式
        self.border_modes = {
            "constant": cv2.BORDER_CONSTANT,  # 常数填充，默认为self.color
            "replicate": cv2.BORDER_REPLICATE,  # 边缘复制
            "reflect": cv2.BORDER_REFLECT,  # 反射填充，以图像边界为对称轴，进行镜像反射。
            "wrap": cv2.BORDER_WRAP,  # 平铺填充，将图像像“卷筒”一样循环平铺
        }
        self.meta = {
            "orig_shape": (0, 0),
            "ratio_x": 1.0,
            "ratio_y": 1.0,
            "pad_info": (0, 0, 0, 0),  # 左右上下边各填充多少
        }

    # 完美继承 cv2.resize
    def resize(self, src, dsize, fx=None, fy=None, interpolation=cv2.INTER_LINEAR):
        res = cv2.resize(src, dsize, fx=fx, fy=fy, interpolation=interpolation)
        h_orig, w_orig = src.shape[:2]
        h_new, w_new = res.shape[:2]
        self.meta["orig_shape"] = (h_orig, w_orig)
        self.meta["ratio_x"] = w_new / w_orig
        self.meta["ratio_y"] = h_new / h_orig
        return res

    # 保持比例缩放
    def ratio_resize(self, img, target_size):
        h, w = img.shape[:2]
        r = min(target_size[0] / h, target_size[1] / w)
        # if not scaleup:
        #     r = min(r, 1.0)
        new_size = (int(round(w * r)), int(round(h * r)))
        return self.resize(img, new_size)

    # 支持多种填充模式的 Pad 函数
    def pad(self, img, target_shape=None, stride=None, mode="constant"):
        """
        以 target_shape 为基准，如果指定了 stride，则填充至满足 stride 的最小倍数（且不小于 target_shape）。
        """
        h, w = img.shape[:2]
        # 1. 确定基础目标宽高 (如果没有提供 target_shape，则默认为原图大小)
        th, tw = target_shape if target_shape else (h, w)
        # 2. 如果指定了 stride，确保目标尺寸是 stride 的倍数
        if stride:
            # math.ceil(target / stride) * stride
            tw = ((tw + stride - 1) // stride) * stride
            th = ((th + stride - 1) // stride) * stride

        # 3. 计算实际需要填充的量 (必须保证填充后不小于原图)
        pad_w = max(0, tw - w)
        pad_h = max(0, th - h)

        # 均分填充
        top, left = pad_h // 2, pad_w // 2
        bottom, right = pad_h - top, pad_w - left

        # 4. 选择填充模式
        border_type = self.border_modes.get(mode, cv2.BORDER_CONSTANT)

        res = cv2.copyMakeBorder(
            img, top, bottom, left, right, borderType=border_type, value=self.color
        )

        # 记录 meta
        self.meta["pad_info"] = (left, right, top, bottom)
        self.meta["target_shape"] = (th, tw)  # 可选：记录实际填充后的尺寸
        return res

    # 裁剪掉填充部分
    def unpad(self, img):
        l, r, t, b = self.meta["pad_info"]

        if img.ndim == 4:
            # 4D Tensor [B, C, H, W]
            return img[:, :, t : img.shape[2] - b, l : img.shape[3] - r]
        elif img.ndim == 3:
            # 判断是 CHW 还是 HWC, 通常 C 很小 (如 1 或 3)，而 H 和 W 较大
            if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
                # 3D Tensor [C, H, W]
                return img[:, t : img.shape[1] - b, l : img.shape[2] - r]
            else:
                # 3D 图像 [H, W, C]
                return img[t : img.shape[0] - b, l : img.shape[1] - r, :]
        else:
            # 2D
            return img[t : img.shape[0] - b, l : img.shape[1] - r]

    # 组合：等比例缩放 + 填充
    def ratio_resize_and_pad(
        self, img, target_shape=(640, 640), stride=None, mode="constant"
    ):
        img_res = self.ratio_resize(img, target_shape)
        return self.pad(img_res, target_shape, stride=stride, mode=mode)

    def scale_coords(self, coords):
        """坐标映射还原"""
        x_pad, y_pad = self.meta["pad"]
        rx, ry = self.meta["ratio"]
        coords[:, [0, 2]] = (coords[:, [0, 2]] - x_pad) / rx
        coords[:, [1, 3]] = (coords[:, [1, 3]] - y_pad) / ry
        h0, w0 = self.meta["orig_shape"]
        coords[:, [0, 2]] = np.clip(coords[:, [0, 2]], 0, w0)
        coords[:, [1, 3]] = np.clip(coords[:, [1, 3]], 0, h0)
        return coords


processor = ODResize()
img = cv2.imread(r"imgs\left.png")
print(img.shape)

# 不等比例缩放，指定目标尺寸
img_processed = processor.resize(img, (640, 640))
print(img_processed.shape)
cv2.imshow("img_processed1", img_processed)

# 不等比例缩放，根据缩放系数缩放
img_processed = processor.resize(img, fx=0.5, fy=0.5, dsize=None)
print(img_processed.shape)
cv2.imshow("img_processed2", img_processed)

# 等比例缩放，也就是最终尺寸不一定满足target_size要求,有一条边是满足的
img_processed = processor.ratio_resize(img, target_size=(640, 1080), scaleup=True)
print(img_processed.shape)
cv2.imshow("img_processed3", img_processed)


# 不缩放图片，仅填充到target_shape尺寸, target_shape小于原图尺寸则不填充
img_processed = processor.pad(img, target_shape=(1000, 1640), mode="replicate")
print(img_processed.shape)
cv2.imshow("img_processed4", img_processed)

# 不缩放图片，仅填充到stride的整数倍
img_processed = processor.pad(img, stride=32, mode="replicate")
print(img_processed.shape)
cv2.imshow("img_processed5", img_processed)

# 等比例缩放 + 填充，满足target_size要求
img_processed = processor.ratio_resize_and_pad(img, (960, 1080), mode="replicate")
print(img_processed.shape)
cv2.imshow("img_processed6", img_processed)

# 等比例缩放 + 填充，满足target_size 和 stride
img_processed = processor.ratio_resize_and_pad(
    img, (961, 1082), stride=32, mode="replicate"
)
print(img_processed.shape)
cv2.imshow("img_processed7", img_processed)


cv2.waitKey(0)

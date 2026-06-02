import os
import random

import cv2
import numpy as np
from tqdm import tqdm

import odpath
from odpath import (
    ODLogger,
    get_files_path,
    get_img_ext,
    show,
    sort_files,
)

if __name__ == "__main__":
    odlog = ODLogger()

    IMG_FORMATS = odpath.get_img_ext()
    odlog.info(IMG_FORMATS)

    imgs_file = "imgs"
    img_paths, img_names = get_files_path(imgs_file)
    random.shuffle(img_paths)
    print(img_paths)
    img_paths = sort_files(img_paths)
    print(img_paths)
    for i, (img_path, img_name) in enumerate(zip(tqdm(img_paths), img_names)):
        if os.path.splitext(img_name)[1] not in IMG_FORMATS:
            continue
        print(f"Progress {i + 1}/{len(img_paths)}: {img_name}")
        image = cv2.imread(img_path)
        print(image.shape)
        show(image)

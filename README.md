参考：https://www.yuque.com/wgd4510/4510/wflwea
## 上传
 先修改setup.py 文件中的版本号等信息
```bash
# 打包生成 tar.gz文件（离线安装）如 dist/odpath-0.0.8.tar.gz
python setup.py sdist 
# 打包生成 whl文件（在线安装）如 dist/odpath-0.0.8-py3-none-any.whl
python setup.py bdist_wheel

# 上传到pypi库，把代理关掉
twine upload dist/odpath-0.0.8-py3-none-any.whl
```


## 安装

```bash
$ pip install odpath
```

## 用法

```python

from odpath import get_files_path
from tqdm import tqdm
import cv2
import os.path as osp

IMG_FORMATS = ['.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm']

imgs_file = r"images"
img_paths, img_names = get_files_path(imgs_file, pattern=[".jpg"], contains=["gray"], onlyfirst=True)
# pattern 指定文件后缀
# contains 指定文件名包含字段
# onlyfirst 指定是否包含子文件夹

for i, (img_path, img_name) in enumerate(zip(tqdm(img_paths), img_names)):
    if osp.splitext(img_name)[1] not in IMG_FORMATS:
        continue
    image = cv2.imread(img_path)
    print(image.shape)

```


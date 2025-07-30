from copy import deepcopy

import cv2
import numpy as np
from ultralytics.utils.instance import Instances

from .augment import Albumentations

def _loadLabelFile(txtPath: str) -> tuple[list[int], list[tuple[float, float, float, float]]]:
  clazzes = []
  bboxes = []
  with open(txtPath, 'r', encoding='utf8') as f:
    for row in f.readlines():
      elements = row.split(' ')
      clazzes.append(int(elements[0]))
      bboxes.append(tuple(map(float, elements[1:])))
  return np.array(clazzes, dtype=np.float32), np.array(bboxes, dtype=np.float32)

def _yolo2xyxy(bbox: tuple[float, float, float, float], imageShape: tuple[int, int]) -> tuple[int, int, int, int]:
  cx, cy, w, h = bbox
  imageHeight = imageShape[0]
  imageWidth = imageShape[1]
  x1 = int((cx - w / 2) * imageWidth)
  y1 = int((cy - h / 2) * imageHeight)
  x2 = int((cx + w / 2) * imageWidth)
  y2 = int((cy + h / 2) * imageHeight)
  return x1, y1, x2, y2

def _printImageLabels(image: np.ndarray, clazzes: np.ndarray, bboxes: np.ndarray):
  print('画像: ', image.shape)
  for clazz, bbox in zip(clazzes, bboxes):
    xyxy = ', '.join(map(str, _yolo2xyxy(bbox, image.shape)))
    print(f'ラベル: {int(clazz)} ({xyxy})')

def _test(imagePath: str, labelPath: str):
  # 元画像読み込み
  oImage = cv2.imread(imagePath)
  # Yoloデータ読み込み
  oClazzes, oBBoxes = _loadLabelFile(labelPath)
  # p=1.0で変換確率を100%にする
  augmenter = Albumentations(p=1.0)

  # 元画像、ラベルの情報を出力
  _printImageLabels(oImage, oClazzes, oBBoxes)
  
  for i in range(100000):
    # 変換前のデータをコピー
    image = deepcopy(oImage)
    bboxes = deepcopy(oBBoxes)
    clazzes = deepcopy(oClazzes)

    # Albumentations で変換する
    transformed = augmenter({
      'img': image,
      'cls': clazzes,
      'instances': Instances(bboxes=bboxes, normalized=True),
    })

    # 返還後の画像、ラベルの情報を出力
    x1, y1, x2, y2 = _yolo2xyxy(transformed['instances'].bboxes[0], transformed['img'].shape)
    for point in [x1, y1, x2, y2]:
      if point < 0 or point > 640:
        _printImageLabels(transformed['img'], transformed['cls'], transformed['instances'].bboxes)


if __name__ == '__main__':
  _test(
    imagePath='data/dataset/images/train/PXL_20250405_021239921_1920_1280_2560_1920.jpg',
    labelPath='data/dataset/labels/train/PXL_20250405_021239921_1920_1280_2560_1920.txt',
  )
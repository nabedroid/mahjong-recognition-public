from typing import Generator

import albumentations as A
import numpy as np

# ラベル名: ID
_label2idDict = {
  '1m': 0, '2m': 1, '3m': 2, '4m': 3, '5m': 4, '6m': 5, '7m': 6, '8m': 7, '9m': 8,
  '1p': 9, '2p': 10, '3p': 11, '4p': 12, '5p': 13, '6p': 14, '7p': 15, '8p': 16, '9p': 17,
  '1s': 18, '2s': 19, '3s': 20, '4s': 21, '5s': 22, '6s': 23, '7s': 24, '8s': 25, '9s': 26,
  'ton': 27, 'nan': 28, 'sha': 29, 'pe': 30,
  'haku': 31, 'hatu': 32, 'chun': 33,    
  'r5m': 34, 'r5p': 35, 'r5s': 36,
}
# ID: ラベル名
_id2labelDict = {value: key for key, value in _label2idDict.items()}

def label2id(label):
  """
  ラベル名をIDに変換する
  """
  return _label2idDict.get(label, None)

def id2label(id):
  """
  IDをラベル名に変換する
  """
  return _id2labelDict.get(id, None)

def _generateGridPoint(image: np.ndarray, cropSize: tuple[int, int], slide: tuple[int, int]):
  """
  画像を切り出す座標を生成する
  """
  imageHeight, imageWidth, _ = image.shape
  for x1 in range(0, imageWidth, slide[0]):
    x1 = min(x1, imageWidth - cropSize[0])
    x2 = x1 + cropSize[0]
    for y1 in range(0, imageHeight, slide[1]):
      y1 = min(y1, imageHeight - cropSize[1])
      y2 = y1 + cropSize[1]
      yield ((x1, y1), (x2, y2))

def generateGridCrops(
    image: np.ndarray,
    labels: list[int],
    bboxes: list[list[int, float, float, float, float]],
    cropSize=(640, 640),
    slide=(320, 320),
    visiblity=0.8,
) -> Generator[tuple[np.ndarray, list[list[int, float, float, float, float]], tuple[tuple[int, int], tuple[int, int]]], None, None]:
  """
  画像を指定範囲で切り出す
  """
  bboxParams = A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=visiblity)
  # 切り出し
  for ((x1, y1), (x2, y2)) in _generateGridPoint(image, cropSize, slide):
    # 画像を指定範囲で切り出し、範囲内のアノテーションデータを抽出
    transform = A.Compose([A.Crop(x_min=x1, y_min=y1, x_max=x2, y_max=y2)], bbox_params=bboxParams)
    transformed = transform(image=image, labels=labels, bboxes=bboxes)
    # labelsとbboxesを結合してYOLO形式のリストに戻す
    yoloTxtRows = [[int(label), *bbox] for label, bbox in zip(transformed['labels'], transformed['bboxes'])]
    yield (
      transformed['image'],
      yoloTxtRows,
      ((x1, y1), (x2, y2)),
    )

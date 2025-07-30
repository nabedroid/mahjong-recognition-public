import glob
import math
import os
import shutil

import cv2

def _loadYoloTxt(txtPath: str) -> list[list[int, float, float, float, float]]:
  if not os.path.exists(txtPath): return []
  with open(txtPath, 'r', encoding='utf8') as f:
      yoloTxtRows = [
        [int(row.split(' ')[0]), *map(float, row.split(' ')[1:])]
        for row in f.readlines()
      ]
  return yoloTxtRows

def _drawRects(image, yoloTxtRows):
  
  for label, cx, cy, width, height in yoloTxtRows:
    imageHeight, imageWidth, _ = image.shape
    x1 = math.floor((cx - width / 2) * imageWidth)
    y1 = math.floor((cy - height / 2) * imageHeight)
    x2 = math.floor((cx + width / 2) * imageWidth)
    y2 = math.floor((cy + height / 2) * imageHeight)
    cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=2)

  return image

if __name__ == '__main__':
  imagePathList = glob.glob('data/dataset/**/*.jpg', recursive=True)

  if os.path.exists('data/draw'):
    shutil.rmtree('data/draw')
  os.makedirs('data/draw')

  for imagePath in imagePathList:
    labelPath = imagePath.replace('images', 'labels').replace('.jpg', '.txt')
    if not os.path.exists(labelPath): continue
    # 画像読み込み
    image = cv2.imread(imagePath)
    # ラベル読み込み
    yoloTxtRows = _loadYoloTxt(labelPath)

    _drawRects(image, yoloTxtRows)

    outPath = os.path.join('data/draw', os.path.basename(imagePath))
    cv2.imwrite(outPath, image)

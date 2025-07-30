"""
labelmeでアノテーションされた画像ファイル及びjsonファイルを、
適切なサイズに分割した後にyolo形式に変換し保存する
"""
import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np

from .labelme2yolo import labelme2yolo
from .labelme_utils import generateGridCrops

def _parseArguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', type=str, default='./data/labelme', help='入力ディレクトリのパス')
  parser.add_argument('--output', type=str, default='./data/dataset', help='出力ディレクトリのパス')
  return parser.parse_args()

def _globLabelmeFiles(labelmeDir: str) -> tuple[list[str], list[str]]:
  """
  指定されたディレクトリ配下に存在するすべてのlabelmeの画像ファイルパス、jsonファイルパスを返す
  """
  result = ([], [])
  jsonNamePathDict = {
    os.path.splitext(os.path.basename(jsonPath))[0]: jsonPath
    for jsonPath in glob.iglob(os.path.join(labelmeDir, '**', '*.json'), recursive=True)
  }

  for filePath in glob.iglob(os.path.join(labelmeDir, '**', '*.*'), recursive=True):
    filename, ext = os.path.splitext(os.path.basename(filePath))
    if ext not in ['.jpg', '.png']: continue
    # 拡張子を除くファイル名
    if filename in jsonNamePathDict:
      result[0].append(jsonNamePathDict[filename])
      result[1].append(filePath)

  return result

def _loadJson(jsonPath: str) -> dict:
  """
  jsonファイルを読み込む
  """
  with open(jsonPath, 'r', encoding='utf8') as f:
    return json.load(f)

def _writeTxt(txtPath: str, txtRows: list[list[int, float, float, float, float]]):
  """
  Yoloのtxt形式のファイルとして書き込む
  """
  # 出力ディレクトリ作成
  os.makedirs(os.path.dirname(txtPath), exist_ok=True)
  with open(txtPath, 'w', encoding='utf8') as f:
    # 各行のリスト（例: [0, 0.5, 0.5, 0.1, 0.1]）をスペース区切りの文字列に変換
    lines = [' '.join(map(str, row)) for row in txtRows]
    f.write('\n'.join(lines))

def _writeImage(image: np.ndarray, imagePath: str):
  """
  画像を書き込む
  """
  os.makedirs(os.path.dirname(imagePath), exist_ok=True)
  cv2.imwrite(imagePath, image)

if __name__ == '__main__':
  args = _parseArguments()

  labelmeDir = args.input
  outputDir = args.output

  # 出力ディレクトリを削除する
  if os.path.exists(outputDir):
    shutil.rmtree(outputDir)

  # Labelmeディレクトリ内から画像とjsonデータをセットで取得する
  jsonPathList, imagePathList  = _globLabelmeFiles(labelmeDir)
  
  # Labelmeの画像とjsonデータをYoloに変換して指定されたディレクトリに出力する
  for i, (imagePath, jsonPath) in enumerate(zip(imagePathList, jsonPathList)):
    # 拡張子を除いたファイル名を取得する
    basename, ext = os.path.splitext(os.path.basename(imagePath))
    # labelme ディレクトリからの相対パス
    relativePath = os.path.relpath(os.path.dirname(imagePath), labelmeDir)
    # 画像読み込み
    image = cv2.imread(imagePath)
    # Yolo形式に変換
    labelmeJson = _loadJson(jsonPath)
    yoloTxtRows = labelme2yolo(labelmeJson)
    labels = list(map(lambda d: d[0], yoloTxtRows))
    bboxes = list(map(lambda d: d[1:], yoloTxtRows))

    # 画像640に切り分け
    for cropImage, cropYoloTxtRows, ((x1, y1), (x2, y2)) in generateGridCrops(image, labels, bboxes):
      # 切り分け画像内に一つも牌が無い場合は飛ばす
      if len(cropYoloTxtRows) == 0: continue
      # 出力先を設定
      cropBasename = f'{basename}_{x1}_{y1}_{x2}_{y2}'
      cropImagePath = os.path.join(outputDir, 'images', relativePath, f'{cropBasename}{ext}')
      cropTxtPath = os.path.join(outputDir, 'labels', relativePath, f'{cropBasename}.txt')
      # 出力
      _writeTxt(cropTxtPath, cropYoloTxtRows)
      _writeImage(cropImage, cropImagePath)


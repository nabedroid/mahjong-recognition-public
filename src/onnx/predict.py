import argparse
import glob
import os
import shutil
import yaml

import onnxruntime
from PIL import Image
from tqdm import tqdm

from .onnx_utils import preprocess, postprocess, drawBoudingBoxes

def _parseArgs():
  """コマンドライン引数を解析します。"""
  parser = argparse.ArgumentParser(description="YOLO ONNXモデルで推論を実行します。")
  parser.add_argument('--model', type=str, required=True, help='ONNXモデルのファイルパス')
  parser.add_argument('--source', type=str, default='data/predict', help='推論対象の画像ファイルまたはディレクトリのパス')
  parser.add_argument('--data', type=str, default='dataset.yaml', help='クラス名が定義されたdataset.yamlのパス')
  parser.add_argument('--conf', type=float, default=0.5, help='信頼度の閾値')
  parser.add_argument('--project', type=str, default='data/runs', help='結果を出力するプロジェクトディレクトリ')
  parser.add_argument('--name', type=str, default='predict', help='結果を出力する実行名（サブディレクトリ）')
  parser.add_argument('--exist-ok', action='store_true', default=False, help='既存の出力ディレクトリを上書きする')

  args = parser.parse_args()

  if not os.path.exists(args.model):
    parser.error(f'モデルファイルが見つかりません: {args.model}')

  return parser.parse_args()

def predict(
    modelPath: str,
    sourcePath: str,
    datasetPath: str,
    conf: float,
    project: str,
    name: str,
    exist_ok: bool,
):
  # モデルを読み込む
  model = onnxruntime.InferenceSession(
    modelPath,
    providers=['CPUExecutionProvider'],
  )

  # 推論対象の画像リストを取得
  imagePathList = [sourcePath]
  if os.path.isdir(sourcePath):
    # ディレクトリが指定された場合
    # ディレクトリ内の画像を全て取得する
    globPath = os.path.join(sourcePath, '**', '*.*')
    imagePathList = [
      filePath for filePath in glob.glob(globPath, recursive=True)
      if os.path.splitext(filePath)[1] in ['.jpg', '.jpeg', '.png']
    ]
  
  # dataset.yaml からIDを取得
  with open(datasetPath, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
  classNames = data['names']

  # 出力先ディレクトリを作成する
  outputDir = os.path.join(project, name)
  if os.path.exists(outputDir):
    if exist_ok:
      shutil.rmtree(outputDir)
    else:
      # 出力先ディレクトリが存在する場合は通番を付与したディレクトリを作成する
      # {project}/{name}{i:0～100}
      for i in range(2, 100):
        outputDir = os.path.join(project, f'{name}{i}')
        if not os.path.exists(outputDir): break
      # 通番が100まで埋まっている場合はエラー
      raise ValueError('outputDir already exists')
  os.makedirs(outputDir, exist_ok=True)

  # 各画像に対して推論を行う
  for imagePath in tqdm(imagePathList, desc="Processing images"):
    image = Image.open(imagePath)
    # 画像を推論可能な形式に変換する
    processedImage, scale, pad = preprocess(image)
    # 推論を実行する
    inputName = model.get_inputs()[0].name
    outputName = model.get_outputs()[0].name
    outputs = model.run([outputName], {inputName: processedImage})[0]
    # 推論結果をBBoxに変換
    bboxList = postprocess(outputs, image.size, scale, pad, conf, 0.5)
    # image に BBox を描画する
    drawImage = drawBoudingBoxes(image, bboxList, classNames)
    # 結果を保存する
    savePath = os.path.join(outputDir, os.path.basename(imagePath))
    drawImage.save(savePath)

if __name__ == "__main__":
  args = _parseArgs()

  predict(
    modelPath=args.model,
    sourcePath=args.source,
    datasetPath=args.data,
    conf=args.conf,
    project=args.project,
    name=args.name,
    exist_ok=args.exist_ok,
  )
"""
画像から牌の検出を行う
"""
import argparse

from ultralytics import YOLO

def _parseArguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default='data/runs/train/weights/best.pt', metavar='PATH', help='学習済みモデルのパス')
  parser.add_argument('--name', type=str, default='predict', metavar='NAME', help='出力先ディレクトリの名前')
  parser.add_argument('--source', type=str, default='data/predict', metavar='PATH', help='検出する画像のディレクトリ、ファイルパス')
  parser.add_argument('--agnostic-nms', action='store_true', help='クラスに依存しないNMSを有効にし、重複するボックスを抑制する')
  
  args = parser.parse_args()
  
  return args

def _predict(modelPath: str, source: str, project: str, name: str, agnostic_nms: bool):
  # Load a model
  model = YOLO(modelPath)
  
  # predict the model
  _ = model.predict(
    source=source,
    project=project,
    name=name,
    agnostic_nms=agnostic_nms,
    save=True,
  )

def _predictAllModel(source: str, agnostic_nms: bool):
  import glob
  import os
  for modelPath in glob.glob('data/runs/**/best.pt', recursive=True):
    name = modelPath.split(os.path.sep)[2]
    _predict(
      modelPath=modelPath,
      source=source,
      project=f'data/runs/{name}',
      name='predict',
      agnostic_nms=agnostic_nms,
    )

if __name__ == '__main__':
  args = _parseArguments()

  if args.model == 'all':
    _predictAllModel(
      source=args.source,
      agnostic_nms=args.agnostic_nms,
    )
  else:
    _predict(
      modelPath=args.model,
      source=args.source,
      project='data/runs',
      name=args.name,
      agnostic_nms=args.agnostic_nms,
    )

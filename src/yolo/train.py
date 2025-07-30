"""
学習を行う
"""
import argparse
import os

from ultralytics import YOLO

def _parseArguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default='yolo11s.pt', metavar='PATH', help='学習済みモデルのパス')
  parser.add_argument('--data', type=str, default='/work/dataset.yaml', metavar='PATH', help='データセット設定ファイルのパス')
  parser.add_argument('--epochs', type=int, default=20, metavar='N', help='学習回数')
  parser.add_argument('--patience', type=int, default=100, metavar='N', help='改善が見られない場合に停止するエポック数')
  parser.add_argument('--batch', type=int, default=8, metavar='N', help='バッチサイズ')
  parser.add_argument('--imgsz', type=int, default=640, metavar='SIZE', help='画像サイズ')
  parser.add_argument('--save-period', type=int, default=-1, metavar='N', help='モデルを保存するエポック数の間隔')
  parser.add_argument('--project', type=str, default='/work/data/runs', metavar='PATH', help='出力先ディレクトリのパス')
  parser.add_argument('--name', type=str, default=None, metavar='NAME', help='トレーニング名、{project}/{name}に結果を出力する')
  parser.add_argument('--exist-ok', action='store_true', help='指定した場合、既存のproject/nameディレクトリを上書きする')
  parser.add_argument('--optimizer', type=str, default='auto', metavar='OPTIMIZER', help='使用するオプティマイザ (SGD, Adam, AdamW, NAdam, RAdam, RMSProp, auto)')
  # ハイパーパラメータ
  parser.add_argument('--lr0', type=float, default=0.01, metavar='LR', help='初期学習率')
  parser.add_argument('--lrf', type=float, default=0.01, metavar='LR', help='最終学習率 (lr0 * lrf)')
  parser.add_argument('--box', type=float, default=7.5, help='box loss gain (バウンディングボックス損失の重み)')
  parser.add_argument('--cls', type=float, default=0.5, help='cls loss gain (クラス分類損失の重み)')
  
  args = parser.parse_args()
  
  return args

def _train(
    modelPath: str,
    datasetPath: str,
    epochs: int,
    patience: int,
    batch: int,
    imgsz: int,
    save_period: int,
    project: str,
    name: str,
    exist_ok: bool,
    optimizer: str,
    lr0: float,
    lrf: float,
    box: float,
    cls: float,
):
  """
  学習を行う
  """
  model = YOLO(modelPath)
  _ = model.train(
    data=datasetPath,
    epochs=epochs,
    patience=patience,
    batch=batch,
    imgsz=imgsz,
    save_period=save_period,
    project=project,
    name=name,
    exist_ok=exist_ok,
    auto_augment=None,
    optimizer=optimizer,
    lr0=lr0,
    lrf=lrf,
    box=box,
    cls=cls,
  )

def _trainMainModels():
  """
  主要なモデル全てに対して学習を行う
  最適な転移学習ベースモデルを見極めるために使用する
  注意: 各バージョンの最上位モデルはGPUメモリの消費が激しい
        epochs: 10, batch: 4
        の設定でもGPUメモリを6GB以上を消費する
        GPUメモリが少ない場合は軽い設定にした方が良い
  """
  models = (
    'yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt',
    'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
    'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt',
  )
  times = {}
  for modelFilename in models:
    # 拡張子を除いたモデル名
    name = os.path.splitext(modelFilename)[0]
    _train(
      modelPath=modelFilename,
      datasetPath='/work/dataset.yaml',
      epochs=10,
      patience=-1,
      batch=2,
      imgsz=640,
      save_period=-1,
      project='/work/data/runs/all',
      name=name,
      exist_ok=True,
      optimizer='auto',
      lr0=0.01,
      lrf=0.01,
      box=7.5,
      cls=0.5,
    )

if __name__ == '__main__':
  args = _parseArguments()

  if args.model == 'all':
    # モデルに'all'が指定された場合
    # 主要モデル全てに学習を行う特別なモードに移行
    _trainMainModels()
  else:
    # 通常の学習を行う
    _train(
      modelPath=args.model,
      datasetPath=args.data,
      epochs=args.epochs,
      patience=args.patience,
      batch=args.batch,
      imgsz=args.imgsz,
      save_period=args.save_period,
      project=args.project,
      name=args.name,
      exist_ok=args.exist_ok,
      optimizer=args.optimizer,
      lr0=args.lr0,
      lrf=args.lrf,
      box=args.box,
      cls=args.cls,
    )

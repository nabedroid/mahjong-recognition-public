from dataclasses import dataclass
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision.ops import nms

@dataclass(frozen=True)
class BoundingBox():
  x1: int
  y1: int
  x2: int
  y2: int
  confidence: float
  classId: int

_CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (255, 165, 0),
    (255, 215, 0), (184, 134, 11), (218, 165, 32), (107, 142, 35), (85, 107, 47),
    (34, 139, 34), (0, 100, 0), (70, 130, 180), (30, 144, 255), (72, 61, 139),
    (255, 192, 203), (255, 20, 147), (138, 43, 226), (165, 42, 42), (210, 105, 30),
    (255, 127, 80), (0, 255, 127), (64, 224, 208), (240, 230, 140), (255, 99, 71),
    (127, 255, 0),
]

def preprocess(image: Image.Image) -> tuple[np.ndarray, float, tuple[int, int]]:
  """Yoloで推論を行う画像の前処理を行う"""
  imageData, scale, pad = letterbox(image, newShape=(640, 640))
  # (H, W, C) -> (C, H, W)
  imageData = imageData.transpose(2, 0, 1)
  # メモリ上で連続した配列にすることで、後の処理を高速化
  imageData = np.ascontiguousarray(imageData)
  imageData = imageData.astype('float32') / 255.
  imageData = np.expand_dims(imageData, axis=0)

  return imageData, scale, pad

def letterbox(
  image: Image.Image,
  newShape: tuple[int, int] = (640, 640),
  color: tuple[int, int, int] = (114, 114, 114)
) -> tuple[np.ndarray, float, tuple[int, int]]:
  """
  アスペクト比を維持したまま画像をリサイズし、指定されたサイズにパディングする (letterbox)。

  Args:
    image (Image.Image): 入力画像 (PIL Image).
    newShape (tuple[int, int]): ターゲットのサイズ (width, height).
    color (tuple[int, int, int]): パディングに使用する色.

  Returns:
    tuple[np.ndarray, float, tuple[int, int]]:
      - パディングされた画像のNumpy配列 (H, W, C).
      - リサイズに使用されたスケール比.
      - パディングのサイズ (padX, padY).
  """
  originalW, originalH = image.size
  targetW, targetH = newShape

  # スケール計算
  scale = min(targetW / originalW, targetH / originalH)
  newW, newH = int(originalW * scale), int(originalH * scale)

  # リサイズ (高品質なバイキュービック法を使用)
  resizedImage = image.resize((newW, newH), Image.Resampling.BICUBIC)

  # パディングして正方形にする
  paddedImage = Image.new('RGB', newShape, color)
  padX = (targetW - newW) // 2
  padY = (targetH - newH) // 2
  paddedImage.paste(resizedImage, (padX, padY))

  return np.array(paddedImage), scale, (padX, padY)

def postprocess(
  outputs: np.ndarray,
  originalImageSize: tuple[int, int],
  scale: float,
  pad: tuple[int, int],
  confThreshold,
  iouThreshold,
) -> list[BoundingBox]:
  """YOLOの出力に対して後処理（NMSなど）を行う"""
  # Numpy配列をTorchテンソルに変換
  predictions = torch.from_numpy(outputs[0])

  # 信頼度の計算: 物体らしさ(obj_conf) * クラス確率(class_prob)
  boxConfidence = predictions[:, 4]
  classProbs = predictions[:, 5:]
  classScores, classIds = torch.max(classProbs, dim=1)
  finalScores = boxConfidence * classScores

  # 信頼度スコアに基づいて候補をフィルタリング
  keep = finalScores > confThreshold
  boxes = predictions[keep, :4]  # フィルタリング後の box (cx, cy, w, h)
  scores = finalScores[keep]
  classIds = classIds[keep]

  # (cx, cy, w, h) -> (x1, y1, x2, y2) に変換
  cx, cy, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
  x1, y1 = cx - width / 2, cy - height / 2
  x2, y2 = cx + width / 2, cy + height / 2
  boxesXyxy = torch.stack((x1, y1, x2, y2), dim=1)
  # torchvision.ops.nms を使用
  indices = nms(boxesXyxy, scores, iouThreshold)

  originalW, originalH = originalImageSize
  padX, padY = pad

  # NMS後の最終的なボックス、スコア、クラスID
  finalBoxes = boxesXyxy[indices]
  finalScores = scores[indices]
  finalClassIds = classIds[indices]

  # パディングを除去し、スケールを元に戻す
  finalBoxes[:, [0, 2]] = (finalBoxes[:, [0, 2]] - padX) / scale
  finalBoxes[:, [1, 3]] = (finalBoxes[:, [1, 3]] - padY) / scale

  # 元の画像の範囲内にクリップ
  finalBoxes[:, [0, 2]] = torch.clamp(finalBoxes[:, [0, 2]], 0, originalW)
  finalBoxes[:, [1, 3]] = torch.clamp(finalBoxes[:, [1, 3]], 0, originalH)

  resultBoxes = []

  for i in range(len(finalBoxes)):
    box = finalBoxes[i].int().tolist()
    resultBoxes.append(BoundingBox(
      x1=box[0], y1=box[1], x2=box[2], y2=box[3],
      confidence=finalScores[i].item(),
      classId=finalClassIds[i].item(),
    ))

  return resultBoxes

def drawBoudingBoxes(
  image: Image.Image,
  bboxes: list[BoundingBox],
  classNames: list[str],
) -> Image.Image:
  """推論結果を画像に描画する"""
  # OpenCV(BGR)に変換
  img = np.array(image)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  for bbox in bboxes:
    # クラスIDに基づいて色を選択（クラス数を超えたら循環させる）
    color = _CLASS_COLORS[bbox.classId % len(_CLASS_COLORS)]
    # RGB -> BGR
    color = (color[2], color[1], color[0])
    # バウンディングボックスを描画
    cv2.rectangle(img, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
    # ラベルを作成
    label = f'{classNames[bbox.classId]}: {bbox.confidence:.2f}'
    # ラベルのテキストサイズを取得
    (labelWidth, labelHeight), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    # ラベルの背景を描画
    label_y1 = max(bbox.y1 - labelHeight - baseline, 0)
    cv2.rectangle(img, (bbox.x1, label_y1), (bbox.x1 + labelWidth, bbox.y1), color, -1)
    # ラベルのテキストを描画（テキスト色は白）
    cv2.putText(img, label, (bbox.x1, bbox.y1 - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

  # OpenCV(BGR)からPIL(RGB)に変換して返す
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return Image.fromarray(img)
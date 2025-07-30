from .labelme_utils import label2id

# labelme形式のアノテーションデータをyolo形式に変換する
def labelme2yolo(json: dict) -> list[list[int, float, float, float, float]]:
  result = []
  # 画像の横幅、高さを取得
  imageWidth = json['imageWidth']
  imageHeight = json['imageHeight']
  # バウンディングボックスごとに処理
  for shape in json['shapes']:
    labelId = label2id(shape['label'])
    # 不明なラベル名は飛ばす
    if labelId is None: continue
    # バウンディングボックス以外はスキップ
    if shape['shape_type'] != 'rectangle': continue
    # 四角の左上、右下の座標
    ((x1, y1), (x2, y2)) = shape['points']
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    # 四角のサイズ
    boxWidth = max(x1, x2) - min(x1, x2)
    boxHeight = max(y1, y2) - min(y1, y2)
    # 座標と幅を正規化して出力する
    result.append([
      labelId, # クラスID
      cx / imageWidth, # 中心座標 x
      cy / imageHeight, # 中心座標 y
      boxWidth / imageWidth, # BBox横幅
      boxHeight / imageHeight, # BBox縦幅
    ])
  return result

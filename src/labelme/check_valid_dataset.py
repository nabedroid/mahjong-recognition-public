import glob
import os




if __name__ == '__main__':
  imageFilePathList = glob.glob('data/dataset/images/**/*.*', recursive=True)
  labelFilePathList = glob.glob('data/dataset/labels/**/*.txt', recursive=True)

  # 無駄なファイルが存在しないか確認
  imageFilePathSet = set(
    os.path.splitext(os.path.relpath(imageFilePath, 'data/dataset/images'))[0]
    for imageFilePath in imageFilePathList
  )
  labelFilePathSet = set(
    os.path.splitext(os.path.relpath(labelFilePath, 'data/dataset/labels'))[0]
    for labelFilePath in labelFilePathList
  )
  print('labelsにしか存在しないファイル:', labelFilePathSet - imageFilePathSet)
  print('imagesにしか存在しないファイル:', imageFilePathSet - labelFilePathSet)
  
  # アノテーションデータが一つもないものがないか確認
  emptyLabelFilePathList = []
  invalidLabelFilePathList = []
  for labelFilePath in labelFilePathList:
    with open(labelFilePath, 'r', encoding='utf8') as f:
      rows = []
      for line in f.readlines():
        elements = line.split(' ')
        rows.append([
          int(elements[0]),
          *[float(p) for p in elements[1:]]
        ])
      if len(rows) == 0:
        emptyLabelFilePathList.append(labelFilePath)
      else:
        # 簡易的にファイル内容をチェックする
        if not all(len(row) == 5 for row in rows):
          invalidLabelFilePathList.append(labelFilePath)
        elif not all(0 <= label <= 36 for label in [row[0] for row in rows]):
          invalidLabelFilePathList.append(labelFilePath)
        elif not all(0 <= p <= 1 for p in [p for row in rows for p in row[1:]]):
          invalidLabelFilePathList.append(labelFilePath)
  print('アノテーションデータが一つもないファイル:', emptyLabelFilePathList)
  print('アノテーションデータが不正なファイル:', invalidLabelFilePathList)

  if len(emptyLabelFilePathList) == 0 and len(invalidLabelFilePathList) == 0:  
    print('OK: データセットは正常です')
  




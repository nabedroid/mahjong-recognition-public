"""
CUDAが有効になっているか確認する
"""
import torch

if __name__ == '__main__':
  print(f"CUDA available: {torch.cuda.is_available()}")
  print(f"Device count: {torch.cuda.device_count()}")
  if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name(0)}")
  
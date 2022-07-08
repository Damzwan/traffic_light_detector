import torch
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
im1 = Image.open('dataset/heon_IMG_0612.JPG')

results = model(im1)

results.print()
results.show()

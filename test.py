from torch import nn
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torch
from PIL import Image

img = read_image('dataset/heon_IMG_0563.JPG')
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
im1 = Image.open('dataset/heon_IMG_0611.JPG')

results = model(im1)

results.print()
results.show()

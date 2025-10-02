# 将训练好的模型导出onnx
from ultralytics import YOLO

model = YOLO('./runs/detect/train/weights/best.pt')
model.export(format="onnx", imgsz=640,simplify=True,nms=True,dynamic=False,opset=21)  # or format="engine"
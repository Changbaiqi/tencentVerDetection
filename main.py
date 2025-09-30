from ultralytics import YOLO
from pathlib import Path

# ---- 1. 加载预训练 YOLOv12n 模型 ----
model = YOLO("yolo12n.pt")  # 也可以换成 yolo12s.pt/yolo12m.pt 等

# ---- 2. 定义数据集 YAML ----
# 假设你之前生成的 data.yaml 路径
data_yaml = "datasets/yolo_dataset/data.yaml"

# ---- 3. 训练模型 ----
# 这里验证码识别使用了generateCapter生成的3500张图片进行，其实训练50轮就已经效果非常好了
results = model.train(
    data=data_yaml,   # 数据集配置文件
    epochs=100,       # 训练轮数
    imgsz=640,        # 输入图片尺寸
    batch=16,         # 可根据显存调节
    device='cpu'          # 0 表示第一块 GPU，改成 'cpu' 用 CPU
)

# ---- 4. 训练完成后推理自己的图片 ----
# 假设你有一张测试图片 test.jpg
test_image = Path("E:\\Yatori-Dev\\tencentImg\\啊边埠_3d135ae7baaebcf54ff2f6047a6fbf7c.png")
results = model.predict(source=test_image, imgsz=640, conf=0.2)  # conf 置信度阈值

# ---- 5. 查看推理结果 ----
# results 是一个列表，每个元素包含预测框、类别、分数等信息
for r in results:
    print(r.boxes.xyxy)     # 预测框 [x1, y1, x2, y2]
    print(r.boxes.conf)     # 置信度
    print(r.boxes.cls)      # 类别 id
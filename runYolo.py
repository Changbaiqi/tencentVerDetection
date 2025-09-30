from ultralytics import YOLO
words =['乘', '仓', '伯', '伴', '佰', '侈', '侧', '倍', '倡', '偿', '傲', '充', '册', '冲', '凹', '办', '北', '匙', '半', '卑', '卜', '卞', '厂', '参', '变', '哺', '啊', '场', '埃', '城', '埠', '堡', '备', '奔', '奥', '安', '宝', '宠', '尘', '尝', '尺', '层', '岔', '岸', '崇', '崩', '差', '巴', '布', '帛', '帮', '常', '弛', '彪', '彻', '忱', '悲', '惩', '惫', '惭', '愁', '憋', '懊', '成', '扁', '才', '扒', '扮', '扳', '把', '报', '抱', '拌', '拔', '拨', '持', '按', '捕', '掣', '搬', '摆', '撤', '播', '敖', '敞', '斑', '昂', '暗', '本', '材', '杯', '板', '柏', '标', '栢', '案', '步', '氨', '池', '沉', '沧', '泊', '波', '泵', '测', '渤', '澄', '澈', '澳', '灿', '炽', '焙', '熬', '爱', '版', '猜', '玻', '班', '瓣', '甭', '畅', '畴', '白', '百', '皑', '盎', '睬', '碍', '碑', '秤', '程', '稠', '笆', '笨', '策', '筹', '箔', '簿', '绊', '绷', '编', '罢', '翅', '翱', '背', '胺', '脖', '膊', '臣', '般', '舱', '舶', '芭', '苍', '苯', '菜', '菠', '蔼', '薄', '虫', '补', '衬', '袄', '裁', '诚', '诧', '豹', '贝', '财', '趁', '跋', '踌', '蹦', '车', '辈', '辨', '辩', '辫', '辰', '边', '迟', '逞', '遍', '邦', '部', '郴', '酬', '钡', '钵', '铂', '长', '阿', '陈', '隘', '雹', '靶', '鞍', '颁', '餐', '饱', '驰', '驳', '骋', '齿']
# 加载训练好的模型
model = YOLO("runs/detect/train2/weights/best.pt")

# 对图片进行检测
results = model("E:\\Yatori-Dev\\tencentImg\\长笆扒_b5db3bea201edb5bb17fda66cf6fe6c8.png")
# results = model("E:\\PycharmProjects\\tencentObjectTrain\\CCC\images\\4.png")
# 获取检测框信息（xyxy坐标, 置信度, 类别）
boxes = results[0].boxes

# 转成 Python 列表 [(置信度, x1, y1, x2, y2, cls), ...]
detections = []
for box in boxes:
    conf = float(box.conf)   # 置信度
    xyxy = box.xyxy.cpu().numpy().flatten().tolist()  # 边框
    cls = int(box.cls)       # 类别ID
    detections.append((conf, *xyxy, cls))

# 按置信度排序
detections = sorted(detections, key=lambda x: x[0], reverse=True)

# 取前3个目标
top3 = detections[:3]

print("Top 3 detections:")
for i, det in enumerate(top3, 1):
    conf, x1, y1, x2, y2, cls = det
    print(f"{i}. Class={cls} [{words[cls]}], Conf={conf:.2f}, Box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

# 如果要在图上只画前3个框：
import cv2
import matplotlib.pyplot as plt

img = results[0].orig_img.copy()
for det in top3:
    conf, x1, y1, x2, y2, cls = det
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
    cv2.putText(img, f"{cls} {conf:.2f}", (int(x1), int(y1)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
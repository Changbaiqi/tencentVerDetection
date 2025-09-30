import json
import os
from pathlib import Path
from shutil import copy2
# 主要用于转换LabelMe标注数据集为yolo训练集

# 输入 LabelMe JSON 的目录
labelmeLabel_dir = "E:\\Yatori-Dev\\tencentData"
labelmeImg_dir = "E:\\Yatori-Dev\\tencentImg"
# 输出 YOLO 数据集目录
output_dir = "datasets/yolo_dataset"
img_out_dir = Path(output_dir) / "train" / "train"
lbl_out_dir = Path(output_dir) / "labels" / "train"

os.makedirs(img_out_dir, exist_ok=True)
os.makedirs(lbl_out_dir, exist_ok=True)

all_classes = set()

# 遍历所有 LabelMe JSON
for json_file in Path(labelmeLabel_dir).rglob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    # 处理 imagePath
    img_path = Path(labelmeImg_dir) / Path(data["imagePath"]).name

    # 拷贝图片
    if img_path.exists():
        copy2(img_path, img_out_dir / img_path.name)

    # YOLO 标签文件
    txt_path = lbl_out_dir / (json_file.stem + ".txt")
    with open(txt_path, "w", encoding="utf-8") as out:
        for shape in data["shapes"]:
            label = shape["label"]
            all_classes.add(label)

            (x1, y1), (x2, y2) = shape["points"]
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            # 转 YOLO 格式（归一化）
            x_center = (x_min + x_max) / 2 / img_w
            y_center = (y_min + y_max) / 2 / img_h
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h

            out.write(f"{label} {x_center} {y_center} {w} {h}\n")

# ---- 构建类别映射 ----
all_classes = sorted(list(all_classes))
class_to_id = {c: i for i, c in enumerate(all_classes)}

# 替换标签里的类别名 → id
for txt_file in lbl_out_dir.rglob("*.txt"):
    lines = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            label = parts[0]
            if label not in class_to_id:
                continue
            cls_id = class_to_id[label]
            new_line = f"{cls_id} " + " ".join(parts[1:])
            lines.append(new_line)
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ---- 生成 data.yaml ----
yaml_path = Path(output_dir) / "data.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(f"train: E:\\PycharmProjects\\tencentObjectTrain\\datasets\\yolo_dataset\\train\\train")
    f.write(f"val: E:\\PycharmProjects\\tencentObjectTrain\\datasets\\yolo_dataset\\train\\train")
    # f.write(f"train: {output_dir}/train/train\n")
    # f.write(f"val: {output_dir}/train/train  # 这里暂时用train当val，建议再划分\n\n")
    f.write(f"nc: {len(all_classes)}\n")
    f.write("names: " + str(all_classes) + "\n")

print("✅ 转换完成！")
print("类别列表:", all_classes)
print("data.yaml 已生成:", yaml_path)

import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
# 程序模仿生成腾讯验证码用于训练

# 配置
OUTPUT_DIR = "datasets"
IMG_SIZE = (672, 480)   # 图像大小
NUM_IMAGES = 3500      # 生成多少张图
CHARS = list("乘仓伯伴佰侈侧倍倡偿傲充册冲凹办北匙半卑卜卞厂参变哺啊场埃城埠堡备奔奥安宝宠尘尝尺层岔岸崇崩差巴布帛帮常弛彪彻忱悲惩惫惭愁憋懊成扁才扒扮扳把报抱拌拔拨持按捕掣搬摆撤播敖敞斑昂暗本材杯板柏标栢案步氨池沉沧泊波泵测渤澄澈澳灿炽焙熬爱版猜玻班瓣甭畅畴白百皑盎睬碍碑秤程稠笆笨策筹箔簿绊绷编罢翅翱背胺脖膊臣般舱舶芭苍苯菜菠蔼薄虫补衬袄裁诚诧豹贝财趁跋踌蹦车辈辨辩辫辰边迟逞遍邦部郴酬钡钵铂长阿陈隘雹靶鞍颁餐饱驰驳骋齿".replace("\n", ""))

# 字体（确保本地有隶书字体）
FONT_PATH = "./兰米汉隶.ttf"  # Windows 下隶书 simli.ttf，Linux/Mac 需放置对应字体文件
FONT_SIZE = 100

def random_background(size):
    """生成随机背景 (纯色 or 噪声)"""
    if random.random() < 0.5:
        # 随机纯色
        color = tuple(random.randint(0, 255) for _ in range(3))
        return Image.new("RGB", size, color)
    else:
        # 随机噪声
        array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        return Image.fromarray(array, "RGB")

def boxes_overlap(box1, box2, margin=10):
    """检查两个 bbox 是否重叠（允许一定间距 margin）"""
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    return not (x2 + margin < a1 or a2 + margin < x1 or
                y2 + margin < b1 or b2 + margin < y1)

def gen_image(img_id):
    img = random_background(IMG_SIZE)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    chars = random.sample(CHARS, 3)  # 随机选 3 个字符
    labels, bboxes = [], []

    for ch in chars:
        for _ in range(100):  # 最多尝试 100 次找位置
            x = random.randint(50, IMG_SIZE[0] - 50)
            y = random.randint(50, IMG_SIZE[1] - 50)

            # 临时画 bbox 计算位置
            bbox = draw.textbbox((x, y), ch, font=font, anchor="mm")

            # 检查是否与已有字符重叠
            if any(boxes_overlap(bbox, b) for b in bboxes):
                continue

            # 通过检查，记录 bbox 并绘制
            draw.text((x, y), ch, font=font, fill=(255, 255, 0), anchor="mm")
            bboxes.append(bbox)

            x1, y1, x2, y2 = bbox
            cls = CHARS.index(ch)
            x_center = ((x1 + x2) / 2) / IMG_SIZE[0]
            y_center = ((y1 + y2) / 2) / IMG_SIZE[1]
            w = (x2 - x1) / IMG_SIZE[0]
            h = (y2 - y1) / IMG_SIZE[1]

            labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            break  # 放置成功
        else:
            print(f"警告: {ch} 在 {img_id} 中放置失败")

    # 保存图片和标签
    img_path = os.path.join(OUTPUT_DIR, "images/train", f"{img_id}.png")
    label_path = os.path.join(OUTPUT_DIR, "labels/train", f"{img_id}.txt")

    img.save(img_path)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

def main():
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

    for i in range(NUM_IMAGES):
        gen_image(i)
        if i % 100 == 0:
            print(f"生成 {i}/{NUM_IMAGES} 张")

if __name__ == "__main__":
    main()
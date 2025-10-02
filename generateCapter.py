import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 程序模仿生成腾讯验证码用于训练

# 配置
OUTPUT_DIR = "datasets/yolo_dataset"
IMG_SIZE = (672, 480)   # 图像大小
NUM_IMAGES = 3500      # 生成多少张图
CHARS = list("乘呈逞才场扳杯摆鞍斥懊熬常笆卑豹澄部长郴崇雹冲仓伴彻侈澈宠称尘卜崩补悲奔迟泵半渤城般北驳绊伯宝瓣泊愁标扮灿睬尝背遍菜餐斑舶报材白憋碍酬策测差佰备蔼澳饱铂保撑本步边布焙搬爱臣碑箔惫罢倡蹦钵充钡驰参侧册匙厂持层舱变绷成啊播稗暗橙班敖畅波车贝倍凹奥胺衬岔玻昂趁苍弛财辰诧诚帮皑苯甭拌炽编偿哺埠拔池秤程堡氨芭辫稠隘傲尺版巴拨簿畴帛把彪筹扁柏跋陈采掣菠卞岸盎安翅百阿抱踌板撤裁辩骋办按忱猜案埃袄膊惭唱敞虫并辨捕蚕邦晨颁承沧沉靶辈薄脖翱惩齿扒".replace("\n", ""))

# 字体（确保本地有隶书字体）
FONT_PATH = "./兰米汉隶.ttf"  # Windows 下隶书 simli.ttf，Linux/Mac 需放置对应字体文件
FONT_SIZE = 100

# 初始化字符计数
char_counts = {ch: 0 for ch in CHARS}

# 噪点背景
def random_noisyPoint_background(size):
    """生成随机彩色背景"""
    array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array, 'RGB')

# 普通渐变
def random_gradient_background(size):
    """生成随机渐变背景"""
    width, height = size
    # 随机选择起始和结束颜色
    start_color = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
    end_color = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)

    # 创建空数组
    array = np.zeros((height, width, 3), dtype=np.uint8)

    # 横向渐变
    for x in range(width):
        t = x / (width - 1)
        color = (1 - t) * start_color + t * end_color
        array[:, x, :] = color.astype(np.uint8)

    return Image.fromarray(array)

# 高频渐变
def high_freq_gradient_background(size, freq=20):
    """
    生成高频彩色渐变背景
    freq: 控制渐变频率，越大变化越快
    """
    width, height = size
    array = np.zeros((height, width, 3), dtype=np.uint8)

    # 随机选择三组渐变频率和相位
    freq_r, freq_g, freq_b = [random.uniform(5, freq) for _ in range(3)]
    phase_r, phase_g, phase_b = [random.uniform(0, 2*np.pi) for _ in range(3)]

    for y in range(height):
        for x in range(width):
            r = 127 * (np.sin(freq_r * x / width * 2*np.pi + phase_r) + 1)
            g = 127 * (np.sin(freq_g * y / height * 2*np.pi + phase_g) + 1)
            b = 127 * (np.sin(freq_b * (x+y) / (width+height) * 2*np.pi + phase_b) + 1)
            array[y, x, :] = [int(r), int(g), int(b)]

    return Image.fromarray(array)



# 随机背景
def random_background(size):
    """
    随机生成背景：
    - 纯色背景 20%
    - 随机噪点背景 20%
    - 线性渐变背景 20%
    - 高频彩色渐变背景 40%
    """
    choice = random.random()


    if choice < 0.2:
        # 纯色背景
        color = tuple(random.randint(0, 255) for _ in range(3))
        return Image.new("RGB", size, color)
    elif choice < 0.4:
        # 随机噪点背景
        return random_noisyPoint_background(size)

    elif choice < 0.6:
        # 线性渐变背景（横向随机渐变）
        return random_gradient_background(size)

    else:
        # 高频彩色渐变背景
        return high_freq_gradient_background(size, freq=10)
# ---------------- 检查 bbox 重叠 ----------------
def boxes_overlap(box1, box2, margin=10):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    return not (x2 + margin < a1 or a2 + margin < x1 or y2 + margin < b1 or b2 + margin < y1)

# ---------------- 生成单张图片 ----------------
def gen_image(img_id):
    img = random_background(IMG_SIZE)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # 选择出现次数最少的字符
    sorted_chars = sorted(CHARS, key=lambda c: char_counts[c])
    chars = sorted_chars[:3]  # 每张图片放 3 个字符
    labels = []
    bboxes = []

    for ch in chars:
        # 获取字符尺寸
        bbox_temp = draw.textbbox((0, 0), ch, font=font, anchor="lt")
        w_char = bbox_temp[2] - bbox_temp[0]
        h_char = bbox_temp[3] - bbox_temp[1]

        for _ in range(100):
            # 保证字符不越界
            x = random.randint(w_char // 2, IMG_SIZE[0] - w_char // 2)
            y = random.randint(h_char // 2, IMG_SIZE[1] - h_char // 2)
            bbox = draw.textbbox((x, y), ch, font=font, anchor="mm")
            if any(boxes_overlap(bbox, b) for b in bboxes):
                continue

            # 绘制字符
            draw.text((x, y), ch, font=font, fill=(255, 255, 0), anchor="mm")
            bboxes.append(bbox)

            # 计算 YOLO 标签
            x1, y1, x2, y2 = bbox
            cls = CHARS.index(ch)
            x_center = ((x1 + x2) / 2) / IMG_SIZE[0]
            y_center = ((y1 + y2) / 2) / IMG_SIZE[1]
            w = (x2 - x1) / IMG_SIZE[0]
            h = (y2 - y1) / IMG_SIZE[1]

            labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            char_counts[ch] += 1
            break
        else:
            print(f"警告: {ch} 在 {img_id} 中放置失败")

    # 创建文件夹并保存
    img_dir = os.path.join(OUTPUT_DIR, "images", "train")
    label_dir = os.path.join(OUTPUT_DIR, "labels", "train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    img.save(os.path.join(img_dir, f"{img_id}.png"))
    with open(os.path.join(label_dir, f"{img_id}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

# ---------------- 主函数 ----------------
def main():
    for i in range(NUM_IMAGES):
        gen_image(i)
        if i % 100 == 0:
            print(f"生成 {i}/{NUM_IMAGES} 张")

if __name__ == "__main__":
    main()
from fontTools.ttLib import TTFont

# 字体文件列表
font_files = [
    r".\兰米汉隶.ttf"
]

# 要检测的字符列表
# chars_to_check = ['以', '我', '这', '和', '何']
chars_to_check = list("乘呈逞才场扳杯摆鞍斥懊熬常笆卑豹澄部长郴崇雹冲仓伴彻侈澈宠称尘卜崩补悲奔迟泵半渤城般北驳绊伯宝瓣泊愁标扮灿睬尝背遍菜餐斑舶报材白憋碍酬策测差佰备蔼澳饱铂保撑本步边布焙搬爱臣碑箔惫罢倡蹦钵充钡驰参侧册匙厂持层舱变绷成啊播稗暗橙班敖畅波车贝倍凹奥胺衬岔玻昂趁苍弛财辰诧诚帮皑苯甭拌炽编偿哺埠拔池秤程堡氨芭辫稠隘傲尺版巴拨簿畴帛把彪筹扁柏跋陈采掣菠卞岸盎安翅百阿抱踌板撤裁辩骋办按忱猜案埃袄膊惭唱敞虫并辨捕蚕邦晨颁承沧沉靶辈薄脖翱惩齿扒".replace("\n", ""))
# 保存没有包含全部字符的字体
missing_fonts = []

for font_path in font_files:
    try:
        font = TTFont(font_path)
        # 获取字体所有字符对应的 Unicode codepoints
        cmap = font['cmap'].getBestCmap()
        font_chars = set(chr(codepoint) for codepoint in cmap.keys())

        # 检查是否包含所有字符
        if not all(c in font_chars for c in chars_to_check):
            missing_fonts.append(font_path)
    except Exception as e:
        print(f"无法读取字体 {font_path}: {e}")
        missing_fonts.append(font_path)

# 输出没有包含全部字符的字体
print("以下字体未包含所有指定字符：")
for f in missing_fonts:
    print(f)

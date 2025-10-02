import cv2
import numpy as np
import onnxruntime as ort

# 加载模型
session = ort.InferenceSession("E:\\PycharmProjects\\tencentObjectTrain\\runs\detect\\train2\weights\\best.onnx", providers=["CUDAExecutionProvider"])
#
# # 预处理输入图像
# def preprocess(image_path):
# image = cv2.imread(image_path)
# resized = cv2.resize(image, (640, 640))
# input_data = resized.transpose(2, 0, 1).astype('float32') / 255.0
# return np.expand_dims(input_data, axis=0), image
#
# # 推理
# def infer(image_path):
# input_tensor, original_image = preprocess(image_path)
# outputs = session.run(None, {"images": input_tensor})
# return outputs, original_image
#
# # 后处理并绘制检测框
# def postprocess(outputs, original_image):
# boxes, scores, class_ids = outputs[0], outputs[1], outputs[2]
# for box, score, class_id in zip(boxes, scores, class_ids):
# if score > 0.5: # 设置置信度阈值
# x1, y1, x2, y2 = map(int, box)
# cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
# cv2.putText(original_image, f"Class {int(class_id)}: {score:.2f}",
# (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
# return original_image
#
# # 执行推理并显示结果
# outputs, original_image = infer("input.jpg")
# result_image = postprocess(outputs[0], original_image)
# cv2.imshow("Result", result_image)
# cv2.waitKey(0)
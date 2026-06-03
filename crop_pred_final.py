import os
import cv2
from ultralytics import YOLO

# 配置
model_path = r"F:\shiyan2-commodity_detect\runs\detect\train_8cls_optimized\weights\best.pt"
image_dir = r"F:\shiyan2-commodity_detect\data\test"
output_dir = r"F:\shiyan2-commodity_detect\cropped_pred_final"
conf_threshold = 0.5
expand_ratio = 0.15

model = YOLO(model_path)

# 清空输出目录，避免旧文件残留
import shutil
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

count = 0
no_detection = 0
processed = set()  # 记录已处理的原始图片名（去重）

for img_name in os.listdir(image_dir):
    if not img_name.endswith('.jpg'):
        continue

    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    results = model(img_path, verbose=False)[0]
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        no_detection += 1
        continue

    best_box = max(boxes, key=lambda x: x.conf[0])
    conf = float(best_box.conf[0])
    if conf < conf_threshold:
        no_detection += 1
        continue

    cls = int(best_box.cls[0])
    pred_class = model.names[cls]

    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

    box_w = x2 - x1
    box_h = y2 - y1
    x1 = max(0, int(x1 - box_w * expand_ratio))
    y1 = max(0, int(y1 - box_h * expand_ratio))
    x2 = min(w, int(x2 + box_w * expand_ratio))
    y2 = min(h, int(y2 + box_h * expand_ratio))

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        continue

    # ✅ 用原始文件名（去掉扩展名）保存，保留 (1) 区分
    base_name = os.path.splitext(img_name)[0]
    save_name = f"{base_name}_pred_{pred_class}_conf{conf:.2f}.jpg"
    cv2.imwrite(os.path.join(output_dir, save_name), roi)
    count += 1

print(f"✅ 裁剪完成！")
print(f"   总图片数: {len(os.listdir(image_dir))}")
print(f"   成功裁剪: {count} 张")
print(f"   未检测到/低置信度: {no_detection} 张")
print(f"   保存路径: {output_dir}")
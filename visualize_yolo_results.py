import os
import cv2
import random
from ultralytics import YOLO

model = YOLO(r'F:\shiyan2-commodity_detect\runs\detect\train_8cls_optimized/weights/best.pt')
image_dir = r'F:\shiyan2-commodity_detect\data\test'
output_dir = r'F:\shiyan2-commodity_detect\yolo_vis'
os.makedirs(output_dir, exist_ok=True)

class_names = ['hks_large', 'hks_small', 'hn_can', 'jlb_can', 'kkkl_can', 'wlj_can', 'xb', 'xb_wt']
sorted_classes = sorted(class_names, key=len, reverse=True)

# 随机选 10 张图片
all_imgs = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
selected = random.sample(all_imgs, min(10, len(all_imgs)))

for img_name in selected:
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)

    # 获取真实类别
    name_clean = img_name.split('(')[0]
    true_class = None
    for cls in sorted_classes:
        if name_clean.startswith(cls):
            true_class = cls
            break

    # 预测
    results = model(img_path, verbose=False)[0]
    boxes = results.boxes

    # 画真实标签
    cv2.putText(img, f"True: {true_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            pred_class = model.names[cls]

            # 画框
            color = (0, 255, 0) if pred_class == true_class else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{pred_class} ({conf:.2f})"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(os.path.join(output_dir, img_name), img)

print(f"可视化结果保存到 {output_dir}")
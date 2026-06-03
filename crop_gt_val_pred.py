import os
import cv2
import shutil

# 配置
image_dir = r"F:\shiyan2-commodity_detect\data\val_images"
output_dir = r"/cropped_gt_val_pred"

class_names = ['hks_large', 'hks_small', 'hn_can', 'jlb_can', 'kkkl_can', 'wlj_can', 'xb', 'xb_wt']

# 清空并创建输出目录
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
for cls_name in class_names:
    os.makedirs(os.path.join(output_dir, cls_name), exist_ok=True)

count = 0

for img_name in os.listdir(image_dir):
    if not img_name.endswith(('.jpg', '.png')):
        continue

    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(image_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        # ✅ 直接用标签文件里的真实类别 ID
        true_cls = int(parts[0])
        true_class = class_names[true_cls]

        # ✅ 直接用标签文件里的边界框坐标
        x_c, y_c, bw, bh = map(float, parts[1:5])
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        base_name = os.path.splitext(img_name)[0]
        save_path = os.path.join(output_dir, true_class, f"{base_name}_gt_{i}.jpg")
        cv2.imwrite(save_path, roi)
        count += 1

print(f"✅ 验证集 GT-ROI 裁剪完成（纯标签）！共 {count} 张")
print(f"   保存路径: {output_dir}")
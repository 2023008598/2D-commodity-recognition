import os
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'F:\shiyan2-commodity_detect\runs\detect\train_8cls_optimized\weights\best.pt')
    image_dir = r'F:\shiyan2-commodity_detect\data\test'

    # 所有 8 个类别（按长度从长到短排序，避免 xb 误匹配 xb_wt）
    all_classes = ['hks_large', 'hks_small', 'hn_can', 'jlb_can', 'kkkl_can', 'wlj_can', 'xb_wt', 'xb']
    all_classes_sorted = sorted(all_classes, key=len, reverse=True)

    total = 0
    correct = 0
    no_detection = 0

    for img_name in os.listdir(image_dir):
        if not img_name.endswith('.jpg'):
            continue

        # 去掉 .jpg 扩展名
        name_without_ext = img_name[:-4]
        # 去掉 (1) 这样的复制标记
        name_clean = name_without_ext.split('(')[0]

        # 匹配真实类别
        true_class = None
        for cls in all_classes_sorted:
            if name_clean.startswith(cls):
                true_class = cls
                break

        if true_class is None:
            print(f"⚠️ 无法识别类别: {img_name}")
            continue

        img_path = os.path.join(image_dir, img_name)

        results = model(img_path, verbose=False)[0]
        boxes = results.boxes

        total += 1

        if boxes is None or len(boxes) == 0:
            no_detection += 1
            continue

        # 取置信度最高的框
        best_box = max(boxes, key=lambda x: x.conf[0])
        pred_class = model.names[int(best_box.cls[0])]

        if pred_class == true_class:
            correct += 1

    print("=" * 50)
    print(f"总图片数: {total}")
    print(f"检测到目标: {total - no_detection}")
    print(f"未检测到: {no_detection}")
    print(f"检测正确: {correct}")
    if total > 0:
        print(f"检测准确率: {correct / total * 100:.2f}%")
        print(f"召回率: {(total - no_detection) / total * 100:.2f}%")
    print("=" * 50)
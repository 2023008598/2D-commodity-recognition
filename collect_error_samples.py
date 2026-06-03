import os
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm

pred_roi_dir = r"F:\shiyan2-commodity_detect\cropped_pred_final"
model_path = r"F:\shiyan2-commodity_detect\best_classifier_new.pth"
error_dir = r"F:\shiyan2-commodity_detect\error_samples"
os.makedirs(error_dir, exist_ok=True)

class_names = ['hks_large', 'hks_small', 'hn_can', 'jlb_can', 'kkkl_can', 'wlj_can', 'xb', 'xb_wt']
sorted_classes = sorted(class_names, key=len, reverse=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('convnextv2_atto', pretrained=False, num_classes=8)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

errors = []
for img_name in os.listdir(pred_roi_dir):
    if not img_name.endswith('.jpg'):
        continue

    name_clean = img_name.split('(')[0]
    true_class = None
    for cls in sorted_classes:
        if name_clean.startswith(cls):
            true_class = cls
            break
    if true_class is None:
        continue

    img_path = os.path.join(pred_roi_dir, img_name)
    try:
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            _, pred_idx = torch.max(outputs, 1)
        pred_class = class_names[pred_idx.item()]

        if pred_class != true_class:
            errors.append((img_name, true_class, pred_class))
    except:
        continue

# 按错误类型分类保存
for img_name, true_cls, pred_cls in errors[:20]:  # 保存前20个错误样本
    error_type = f"{true_cls}_as_{pred_cls}"
    os.makedirs(os.path.join(error_dir, error_type), exist_ok=True)
    src = os.path.join(pred_roi_dir, img_name)
    dst = os.path.join(error_dir, error_type, img_name)
    shutil.copy2(src, dst)

print(f"收集了 {len(errors)} 个错误样本，已保存到 {error_dir}")
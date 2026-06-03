import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 配置
pred_roi_dir = r"F:\shiyan2-commodity_detect\cropped_pred_final"
model_path = r"F:\shiyan2-commodity_detect\best_classifier_new.pth"
class_names = ['hks_large', 'hks_small', 'hn_can', 'jlb_can', 'kkkl_can', 'wlj_can', 'xb', 'xb_wt']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = timm.create_model('convnextv2_atto', pretrained=False, num_classes=8)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

all_preds = []
all_labels = []

sorted_classes = sorted(class_names, key=len, reverse=True)

for img_name in os.listdir(pred_roi_dir):
    if not img_name.endswith('.jpg'):
        continue

    # 解析真实类别
    name_clean = img_name.split('(')[0]
    true_class = None
    for cls in sorted_classes:
        if name_clean.startswith(cls):
            true_class = cls
            break
    if true_class is None:
        continue

    true_idx = class_names.index(true_class)

    # 预测
    img_path = os.path.join(pred_roi_dir, img_name)
    try:
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            _, pred_idx = torch.max(outputs, 1)

        all_labels.append(true_idx)
        all_preds.append(pred_idx.item())
    except:
        continue

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.title('End-to-End - Pred-ROI Confusion Matrix (Normalized)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(r'F:\shiyan2-commodity_detect\confusion_matrix_pred.png', dpi=300)
plt.show()

print("\n" + "=" * 60)
print("Pred-ROI (End-to-End) Classification Report")
print("=" * 60)
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 配置
data_path = r"F:\shiyan2-commodity_detect\cls_data_new"
model_path = r"F:\shiyan2-commodity_detect\best_classifier_new.pth"
class_names = ['hks_large', 'hks_small', 'hn_can', 'jlb_can', 'kkkl_can', 'wlj_can', 'xb', 'xb_wt']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = timm.create_model('convnextv2_atto', pretrained=False, num_classes=8)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载验证集
val_dataset = datasets.ImageFolder(r'F:\shiyan2-commodity_detect\cls_data_new\val', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# 收集预测和真实标签
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# 绘图
plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.title('ConvNeXt V2 - GT-ROI Confusion Matrix (Normalized)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(r'F:\shiyan2-commodity_detect\confusion_matrix_gt.png', dpi=300)
plt.show()

# 打印分类报告
print("\n" + "="*60)
print("GT-ROI Classification Report")
print("="*60)
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
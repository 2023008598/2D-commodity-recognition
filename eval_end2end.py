import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import torchvision.datasets as datasets

# 配置路径
pred_roi_dir = r"F:\shiyan2-commodity_detect\cropped_pred_final"
model_path = r"F:\shiyan2-commodity_detect\best_classifier_new.pth"

# ✅ 从训练数据获取正确的类别顺序
data_path = r"F:\shiyan2-commodity_detect\cls_data_new\train"
temp_dataset = datasets.ImageFolder(data_path)
class_names = temp_dataset.classes
print(f"类别顺序: {class_names}")

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = timm.create_model('convnextv2_atto', pretrained=False, num_classes=8)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 按长度倒序排序，用于匹配文件名
sorted_classes = sorted(class_names, key=len, reverse=True)

total = 0
correct = 0

print("开始端到端评估...")
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

    # 分类器预测
    img_path = os.path.join(pred_roi_dir, img_name)
    try:
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            pred_class = class_names[predicted.item()]

        total += 1
        if pred_class == true_class:
            correct += 1
    except Exception as e:
        pass

print("\n" + "=" * 50)
print(f"端到端评估完成！")
print(f"总测试图片数: {total}")
print(f"分类正确数: {correct}")
print(f"端到端准确率: {correct / total * 100:.2f}%")
print("=" * 50)
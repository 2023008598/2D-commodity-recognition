import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 配置
    data_path = r"F:\shiyan2-commodity_detect\cls_data_new"
    batch_size = 32
    epochs = 30
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据增强
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据
    train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(os.path.join(data_path, 'val'), transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"类别数: {len(train_dataset.classes)}")
    print(f"类别名: {train_dataset.classes}")
    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")

    # 模型
    model = timm.create_model('convnextv2_atto', pretrained=True, num_classes=8)
    model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 记录训练曲线
    train_losses = []
    val_accs = []
    best_acc = 0

    # 训练
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        val_accs.append(acc)
        print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Acc: {acc:.2f}%')

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), r'F:\shiyan2-commodity_detect\best_classifier_new.pth')

    print(f'✅ 训练完成！最佳准确率: {best_acc:.2f}%')

    # 绘制训练曲线
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(train_losses, 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(val_accs, 'r-', label='Val Accuracy')
    ax2.set_ylabel('Accuracy (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('ConvNeXt V2 Training Curves')
    fig.legend(loc='upper right')
    plt.savefig(r'F:\shiyan2-commodity_detect\training_curves.png', dpi=300)
    plt.show()
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11l.pt')

    results = model.train(
        data='data.yaml',
        epochs=80,
        imgsz=1280,
        batch=6,  # 保守一点
        device=0,
        workers=2,  # 降低 worker 数，省内存
        amp=True,
        cache=False,  # 关闭缓存，省内存
        patience=20,

        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        scale=0.5,

        name='train_8cls_optimized'
    )

    print("训练完成！")
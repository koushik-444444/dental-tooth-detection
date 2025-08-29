import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model for dental tooth detection')
    parser.add_argument('--data', type=str, default='data.yaml', help='Dataset configuration file')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Model variant (yolov8s.pt, yolov8m.pt, yolov8l.pt)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project directory')
    parser.add_argument('--name', type=str, default='dental_detection', help='Experiment name')
    return parser.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    
    # Train the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=50,
        save=True,
        plots=True,
        val=True,
        verbose=True,
        # Enhanced training parameters
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0
    )

if __name__ == '__main__':
    main()

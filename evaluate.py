import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model for dental tooth detection')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, default='data.yaml', help='Dataset configuration file')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save-json', action='store_true', help='Save results to JSON')
    return parser.parse_args()

def main():
    args = parse_args()
    
    
    # Load trained model
    model = YOLO(args.model)
    
    # Run evaluation
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        save_json=args.save_json,
        verbose=True
    )
    
    # Print detailed results
    print("\n EVALUATION RESULTS")
    print("=" * 30)
    print(f" Precision (P): {metrics.box.mp:.4f}")
    print(f" Recall (R): {metrics.box.mr:.4f}")
    print(f" mAP@0.5: {metrics.box.map50:.4f}")
    print(f" mAP@0.5:0.95: {metrics.box.map:.4f}")
    
    # Calculate F1 score
    precision = metrics.box.mp
    recall = metrics.box.mr
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"🔢 F1-Score: {f1_score:.4f}")
    
    return metrics

if __name__ == '__main__':
    main()

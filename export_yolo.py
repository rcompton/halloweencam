from ultralytics import YOLO

def main():
    """
    Export a YOLOv8 model to TensorRT format.
    """
    model = YOLO("yolov8n-seg.pt")
    model.export(format="engine")

if __name__ == "__main__":
    main()

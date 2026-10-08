from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model_seg = YOLO('yolov8s-seg.pt')
model_pose = YOLO('yolov8s-pose.pt')
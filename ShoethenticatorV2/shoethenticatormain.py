from ultralytics import YOLO
import os

model = YOLO('yolov8n.yaml')

model.load('yolov8n.pt')

data_path = r'C:\Users\User\Desktop\ShoethenticatorV2\data.yaml'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"'{data_path}' does not exist")

train_results = model.train(data=data_path, epochs=10)

val_results = model.val()

image_path = r'C:\Users\User\Pictures\j1_ex1.png'

if os.path.exists(image_path):
    try:
        image_results = model(image_path)
        print("Image processing successful.")
    except Exception as e:
        print(f"Error processing image: {e}")
else:
    print(f"Image path does not exist: {image_path}")

model.export(format='onnx')
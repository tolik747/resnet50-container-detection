# import cv2
# import torch
# import torchvision.transforms as transforms
# from torchvision import models
# import torch.nn as nn
# import numpy as np
# from flask import Flask, render_template, Response

# app = Flask(__name__)

# # Завантаження моделі
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# model.fc = nn.Linear(model.fc.in_features, 11)
# model.load_state_dict(torch.load('best_resnet50_model.pth', map_location=device))
# model.to(device).eval()

# # Класи
# classes = ['airplane', 'automobile', 'bird', 'cat', 'container', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck']

# # Трансформація
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# def generate_frames():
#     cap = cv2.VideoCapture(0)
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         # Обробка контурів
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         for cnt in contours:
#             x, y, w, h = cv2.boundingRect(cnt)

#             if 100 < w < 400 and 100 < h < 400:
#                 roi = frame[y:y+h, x:x+w]

#                 try:
#                     img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#                     input_tensor = transform(img).unsqueeze(0).to(device)

#                     with torch.no_grad():
#                         output = model(input_tensor)
#                         pred_class = classes[torch.argmax(output).item()]

#                     if pred_class == 'container':
#                         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                         cv2.putText(frame, f'{pred_class}', (x, y - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                 except:
#                     continue

#         # Кодування кадру
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video')
# def video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5055, debug=False)
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO
import time

app = Flask(__name__)

# load resnet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 11)
resnet_model.load_state_dict(torch.load('best_resnet50_model.pth', map_location=device))
resnet_model.to(device).eval()

# Завантаження YOLOv5
yolo_model = YOLO('yolov5n.pt') #or yolov5s.pt

# klass
classes = ['airplane', 'automobile', 'bird', 'cat', 'container', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Трансформація для resnet
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    persistent_detections = []
    detection_timeout = 10

    while True:
        success, frame = cap.read()
        if not success:
            break

        display_frame = frame.copy()
        results = yolo_model(frame)[0]
        current_detections = []

        if results.boxes:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]

                if roi.size == 0:
                    continue

                try:
                    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    input_tensor = transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = resnet_model(input_tensor)
                        confidence, idx = torch.max(torch.softmax(output, dim=1), 1)
                        pred_class = classes[idx.item()]

                    if confidence.item() > 0.85:
                        box_dims = (x1, y1, x2 - x1, y2 - y1)
                        is_duplicate = any(iou(box_dims, pd[0]) > 0.5 for pd in persistent_detections)
                        if not is_duplicate:
                            current_detections.append((box_dims, pred_class, confidence.item()))
                except:
                    continue
        else:
            # Якщо YOLO нічого не знайшов — класифікуй весь кадр
            try:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = resnet_model(input_tensor)
                    confidence, idx = torch.max(torch.softmax(output, dim=1), 1)
                    pred_class = classes[idx.item()]

                if confidence.item() > 0.85:
                    current_detections.append(((50, 50, 300, 300), pred_class, confidence.item()))
            except:
                pass

        # Оновлення persistent_detections
        updated_persistent = []
        for det in persistent_detections:
            bbox, cls, conf, ttl = det
            ttl -= 1
            if ttl > 0:
                updated_persistent.append((bbox, cls, conf, ttl))

        for new_det in current_detections:
            updated_persistent.append((new_det[0], new_det[1], new_det[2], detection_timeout))

        persistent_detections = updated_persistent

        for det in persistent_detections:
            (x, y, w, h), cls, conf, ttl = det
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, f'{cls} ({conf:.2f})',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)

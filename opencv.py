# import cv2
# import torch
# import torchvision.transforms as transforms
# from torchvision import models
# import torch.nn as nn
# import numpy as np

# # Пристрій
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Завантаження моделі
# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# model.fc = nn.Linear(model.fc.in_features, 11)
# model.load_state_dict(torch.load('host/best_resnet50_model.pth', map_location=device))
# model.to(device).eval()

# # Класи
# classes = ['airplane', 'automobile', 'bird', 'cat', 'container', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck']

# # Трансформація для ResNet-50
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# # Ініціалізація Selective Search
# ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# # Камера
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)  # ширина
# cap.set(4, 720)   # висота

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Selective Search
#     ss.setBaseImage(frame)
#     ss.switchToSelectiveSearchFast()
#     rects = ss.process()

#     # Обмеження кількості регіонів (інакше FPS падає)
#     for (x, y, w, h) in rects[:100]:
#         if w < 80 or h < 80 or w > 400 or h > 400:
#             continue  # пропускаємо дуже малі або великі

#         roi = frame[y:y+h, x:x+w]
#         try:
#             input_tensor = transform(roi).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 output = model(input_tensor)
#                 pred_class = classes[torch.argmax(output).item()]

#             if pred_class == 'container':
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 cv2.putText(frame, f'{pred_class}', (x, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#         except:
#             continue

#     # Показ кадру
#     cv2.imshow('Container Detection (ResNet-50 + Selective Search)', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 11)
model.load_state_dict(torch.load('host/best_resnet50_model.pth', map_location=device))
model = model.to(device)
model.eval()

#clases
classes = ['airplane', 'automobile', 'bird', 'cat', 'container', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ESP32-CAM
#teraz vyuzivane kameru PC
cap = cv2.VideoCapture(0)  # nas ip kamery

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    
    input_tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = classes[torch.argmax(output).item()]

    
    cv2.putText(frame, f'Class: {pred_class}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ESP32-CAM Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
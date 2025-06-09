import os
from torchvision import transforms
from PIL import Image
import random


container_folder = './dataset/train/container'
output_folder = './dataset/train/container'  

#  трансформації
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
])

# скфльки фото буде з одеого
copies_per_image = 10


container_images = [img for img in os.listdir(container_folder) if img.lower().endswith(('png', 'jpg', 'jpeg'))]

print(f'Знайдено {len(container_images)} оригінальних фото контейнерів')

# беремо кожне фото
for img_name in container_images:
    img_path = os.path.join(container_folder, img_name)
    original_image = Image.open(img_path).convert('RGB')

    # нові фото
    for i in range(copies_per_image):
        augmented_image = augmentation(original_image)
        save_path = os.path.join(container_folder, f'{os.path.splitext(img_name)[0]}_aug_{i}.png')
        augmented_image.save(save_path)

print('урааа фініш')

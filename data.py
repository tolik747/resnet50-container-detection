import os
import shutil
from torchvision import datasets
from torchvision.transforms import ToPILImage
import random
from tqdm import tqdm


def create_folder_structure(base_path, classes):
    for split in ['train', 'val', 'test']:
        for class_name in classes:
            os.makedirs(os.path.join(base_path, split, class_name), exist_ok=True)


def save_cifar10_to_folders(dataset, base_path, split_ratios, classes_map):
    data_by_class = {i: [] for i in range(10)}
    
   
    for img, label in tqdm(dataset, desc="rozdelenie CIFAR-10 class"):
        data_by_class[label].append(img)

    # Збереження у відповідні папки
    for class_idx, images in data_by_class.items():
        random.shuffle(images)
        n_total = len(images)
        n_train = int(split_ratios[0] * n_total)
        n_val = int(split_ratios[1] * n_total)
        
        for idx, img in enumerate(images):
            if idx < n_train:
                split = 'train'
            elif idx < n_train + n_val:
                split = 'val'
            else:
                split = 'test'

            class_name = classes_map[class_idx]
            
            img.save(os.path.join(base_path, split, class_name, f'{class_name}_{idx}.png'))

# copy image in folder
def split_container_images(container_folder, base_path, split_ratios):
    all_images = os.listdir(container_folder)
    random.shuffle(all_images)
    n_total = len(all_images)
    n_train = int(split_ratios[0] * n_total)
    n_val = int(split_ratios[1] * n_total)

    for idx, img_name in enumerate(all_images):
        img_path = os.path.join(container_folder, img_name)

        if idx < n_train:
            split = 'train'
        elif idx < n_train + n_val:
            split = 'val'
        else:
            split = 'test'

        os.makedirs(os.path.join(base_path, split, 'container'), exist_ok=True)
        shutil.copy(img_path, os.path.join(base_path, split, 'container', img_name))

# main function
if __name__ == "__main__":
    base_path = './dataset'  # Папка, куди все зберігаємо
    container_images_path = './Prekladisko'  # conteiner photo

    # class CIFAR-10
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']

    # load CIFAR-10
    train_cifar = datasets.CIFAR10(root='./data', train=True, download=True)
    test_cifar = datasets.CIFAR10(root='./data', train=False, download=True)

    full_cifar = train_cifar + test_cifar

    # structure files
    create_folder_structure(base_path, cifar10_classes + ['container'])

  
    save_cifar10_to_folders(full_cifar, base_path, split_ratios=[0.7, 0.15, 0.15],
                            classes_map={i: name for i, name in enumerate(cifar10_classes)})

    
    split_container_images(container_images_path, base_path, split_ratios=[0.7, 0.15, 0.15])

    print("\nfinish")



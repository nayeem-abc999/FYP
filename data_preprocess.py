import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from utils import create_directory
from albumentations import HorizontalFlip, VerticalFlip, Rotate


# Data Augmentation function
def augument_data(images, masks, save_path, augument=True):

    # Define image size
    size = (512, 512)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        # Extracting the name of the file
        name = os.path.basename(x).split(".")[0]  # Cross-platform compatible

        # Reading image and mask
        x = cv2.imread(x, cv2.IMREAD_COLOR)  # Read JPG image
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  # Read TIFF mask

        # Handle augmentation
        if augument:
            # Horizontal Flip
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=np.expand_dims(y, axis=-1))
            x1 = augmented["image"]
            y1 = augmented["mask"][:, :, 0]

            # Vertical Flip
            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=np.expand_dims(y, axis=-1))
            x2 = augmented["image"]
            y2 = augmented["mask"][:, :, 0]

            # Rotate
            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=np.expand_dims(y, axis=-1))
            x3 = augmented["image"]
            y3 = augmented["mask"][:, :, 0]

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]
        else:
            X = [x]
            Y = [y]

        # Save augmented images and masks
        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            temp_image_name = f"{name}_{index}.png"
            temp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "images", temp_image_name)
            mask_path = os.path.join(save_path, "masks", temp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

# Load dataset paths
def load_data(path):
    train_X = sorted(glob(os.path.join(path, "train", "images", "*.png")))  
    train_y = sorted(glob(os.path.join(path, "train", "masks", "*.png")))  

    test_X = sorted(glob(os.path.join(path, "test", "images", "*.png"))) 
    test_y = sorted(glob(os.path.join(path, "test", "masks", "*.png"))) 

    return (train_X, train_y), (test_X, test_y)

if __name__ == "__main__":
    # SEEDING
    np.random.seed(42)

    # Load the data
    data_path = "./dataset"
    (train_X, train_y), (test_X, test_y) = load_data(data_path)
    
    # Print the number of training and testing samples
    print(f"Train: {len(train_X)} - {len(train_y)}")
    print(f"Test: {len(test_X)} - {len(test_y)}")

    # Create directories
    create_directory("final_dataset/train/images/")
    create_directory("final_dataset/train/masks/")
    create_directory("final_dataset/test/images/")
    create_directory("final_dataset/test/masks/")

    # Perform data augmentation
    augument_data(train_X, train_y, "final_dataset/train/", augument=True)
    augument_data(test_X, test_y, "final_dataset/test/", augument=False)

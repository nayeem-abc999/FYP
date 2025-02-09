import os
import time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from model import UNet
from utils import create_directory, seeding

def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred, zero_division=0) 
    score_f1 = f1_score(y_true, y_pred, zero_division=0)
    score_recall = recall_score(y_true, y_pred, zero_division=0)
    score_precision = precision_score(y_true, y_pred, zero_division=0)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_directory("results")

    """ Load dataset """
    test_x = sorted(glob("final_dataset/test/images/*"))
    test_y = sorted(glob("final_dataset/test/masks/*"))

    """ Hyperparameters """
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = "saved_models/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        name = x.split(os.sep)[-1].split(".")[0]

        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
        x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
        x = x / 255.0
        x = np.expand_dims(x, axis=0).astype(np.float32)  ## (1, 3, 512, 512)
        x = torch.from_numpy(x).to(device)

        """ Reading mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
        y = np.expand_dims(mask, axis=0) / 255.0
        y = np.expand_dims(y, axis=0).astype(np.float32)  ## (1, 1, 512, 512)
        y = torch.from_numpy(y).to(device)

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))

            pred_y = pred_y[0].cpu().numpy().squeeze() > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate(
            [image, line, ori_mask, line, pred_y * 255], axis=1
        )
        cat_images /= np.max(cat_images)
        plt.imsave(os.path.join('results', f'{name}.png'), cat_images)
  
    jaccard = metrics_score[0] / len(test_x)
    f1 = metrics_score[1] / len(test_x)
    recall = metrics_score[2] / len(test_x)
    precision = metrics_score[3] / len(test_x)
    acc = metrics_score[4] / len(test_x)
    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")

    fps = 1 / np.mean(time_taken)
    print("FPS: ", fps)

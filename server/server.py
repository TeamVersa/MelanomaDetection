from flask import Flask, jsonify, request
import time

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import albumentations as A
import geffnet
import csv
from PIL import Image

device = torch.device('cpu')

# parameters
model_dir = 'model'
kernel_type = '9c_b7ns_1e_640_ext_15ep'
enet_type = 'efficientnet-b7'
out_dim = 9
image_size = 640

# the model will inference the image num_folds * num_transforms times. This can significantly increase inference times
num_folds, num_transforms = 2, 1 #5, 8


# dataset
class SIIMISICDataset(Dataset):
    def __init__(self, csv, split, mode, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.split = split
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = image[:, :, ::-1]

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        if self.mode == 'test':
            return torch.tensor(image).float()
        else:
            return torch.tensor(image).float(), torch.tensor(self.csv.iloc[index].target).long()


class SingleImage(Dataset):
    def __init__(self, image, transform=None):
        self.image = image
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        image = self.image[:, :, ::-1]

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        return torch.tensor(image).float()


# model architecture
class enetv2(nn.Module):
    def __init__(self, backbone, out_dim, n_meta_features=0, load_pretrained=False):

        super(enetv2, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = geffnet.create_model(
            enet_type.replace('-', '_'), pretrained=load_pretrained)
        self.dropout = nn.Dropout(0.5)

        in_ch = self.enet.classifier.in_features
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        x = self.myfc(self.dropout(x))
        return x

# image transform
def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def load_models(n_fold=5):
    models = []
    for i_fold in range(n_fold):
        model = enetv2(enet_type, n_meta_features=0, out_dim=out_dim)
        model = model.to(device)
        model_file = os.path.join(
            model_dir, f'{kernel_type}_best_fold{i_fold}.pth')
        state_dict = torch.load(model_file, map_location=device)
        state_dict = {k.replace('module.', '')
                                : state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        models.append(model)

    return models


def predict(models, cv2_image, n_test=8):
    transforms_val = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize()
    ])

    image = SingleImage(cv2_image, transform=transforms_val)[0]
    # a single image need to be added a new axis to act like batch_size = 1
    image = image.to(device).unsqueeze(0)

    with torch.no_grad():
        probs = torch.zeros((image.shape[0], out_dim)).to(device)
        for model in models:
            for I in range(n_test):
                l = model(get_trans(image, I))
                probs += l.softmax(1)
                print('#', end='')
        print('')
    probs /= len(models) * n_test

    prediction = probs[:, 6].item()
    return prediction


def crop(image):
    h, w, _ = image.shape
    s = min(w, h)
    x, y = (w-s)//2, (h-s)//2
    cropped = image[y:y+s, x:x+s]
    cv2.imwrite('image_cropped.jpg', cropped)
    return cropped




app = Flask(__name__)

models = load_models(num_folds)


@app.route('/melanoma', methods=['POST'])
def get_tasks():
    print(f'received request: {request}')
    request.files.get('image', '').save('image.jpg')
    image = cv2.imread('image.jpg')
    cropped = crop(image)

    result = predict(models, cropped, num_transforms)

    print(f'result: {result}')
    return jsonify({
        'confidence': result
    })


if __name__ == '__main__':
    print('hello world!')
    app.run(debug=True, host='0.0.0.0')

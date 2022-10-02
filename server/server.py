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

transforms_val = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize()
])

# model architecture
class enetv2(nn.Module):
    def __init__(self, backbone, out_dim, n_meta_features=0, load_pretrained=False):

        super(enetv2, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = geffnet.create_model(enet_type.replace('-', '_'), pretrained=load_pretrained)
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


# load models
models = []
for i_fold in range(5):
    model = enetv2(enet_type, n_meta_features=0, out_dim=out_dim)
    model = model.to(device)
    model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{i_fold}.pth')
    state_dict = torch.load(model_file, map_location=device)
    state_dict = {k.replace('module.', ''): state_dict[k] for k in state_dict.keys()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    models.append(model)
len(models)

# image transform
def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2,3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


# load inference data
# patient 1 image name
in1 = 'hello'
# patient 1 patient_id
pi1 = 'IP23232'
# patient 1 sex
s1 = 'male'
# patient 1 age
a1 = 70.0
# patient 1 anatom_site_general_challenge. where its located
asgc1 = 'torso'
# patient 1 image filepath
f1 = 'image_cropped.jpg'
with open('datasettesting.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "patient_id", "sex", "age_approx", "anatom_site_general_challenge", "width", "height", "filepath"])
    writer.writerow([in1,pi1, s1, a1, asgc1, 6000, 4000, f1])


df_single_image = pd.read_csv('datasettesting.csv')




#n_test = 8
n_test = 1

print('running model...')

def run_ensemble():
    dataset_test = SIIMISICDataset(df_single_image, 'test', 'test', transform=transforms_val)
    image = dataset_test[0]  
    image = image.to(device).unsqueeze(0)  # a single image need to be added a new axis to act like batch_size = 1

    with torch.no_grad():
        probs = torch.zeros((image.shape[0], out_dim)).to(device)
        for model in models:
            for I in range(n_test):
                l = model(get_trans(image, I))
                probs += l.softmax(1)
                print('#', end='')
    probs /= len(models) * n_test

    prediction = probs[:, 6].item()
    return prediction


def crop():
    with Image.open('image.jpg') as image:
        width, height = image.size
        a, b = min(width, height), max(width, height)
        padding = (b - a) / 2
        test_image = image.crop((padding, 0, b-padding, a))
        test_image.save('image_cropped.jpg', "JPEG")


def predict():
    prediction = run_ensemble()
    result = {
        'prediction': 'Benign' if prediction < 0.5 else 'Malignant',
        'confidence': prediction
    }

    return result


app = Flask(__name__)


@app.route('/melanoma', methods=['POST'])
def get_tasks():
    i = request.files.get('image', '')
    print(f'received image {i}')
    i.save('image.jpg')
    crop()

    result = predict()

    print(result)
    return jsonify(result)


if __name__ == '__main__':
    print('hello world!')
    app.run(debug=True, host='0.0.0.0')

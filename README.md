# MelaScan — Melanoma detection in a snap!

Melanoma, the deadliest type of skin cancer, is actually easily treatable if detected at an early stage. However, due to how easy it is to overlook on the skin and how difficult it is to recognize by non-specialized doctors, it often goes undetected until it's too late. Our goal is to build a decision support app to help doctors in the **diagnosis of melanoma in a very easy to use and rapid fashion**, <30 seconds from launch to analysis result.

The MelaScan app takes a photo of the mole on the patient's skin. With the help of our AI model, it instantly returns a probability of this mole being a case of melanoma.
![](https://d112y698adiu2z.cloudfront.net/photos/production/software_photos/002/240/178/datas/gallery.jpg)

## Tech Stack
Cross-platform app: Flutter

RESTful API local server: Flask on Python

Image analysis: PyTorch (Ensemble of EfficientNets)

## Limitations
One of the biggest challenges we ran into was the **extremely imbalanced dataset**. Only 1.76% of the whole dataset contains positive examples (melanoma). Another problem also related to the data is the lack of variety, especially little data on people of color, potentially leading to poor performance and **bias against minorities**.
More metadata in the future could increase accuracy: e.g. exposure for UV-Radiation, usage of skin care products, a marker for the size of the mole, hereditary and nodule mole type.

## Installation
Clone the repository on the server, download the pretrained models from the link in server/models/readme.txt, then run the server: `python server.py`

Flutter app can be installed onto device as normal, precompiled binaries available soon.

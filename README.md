# MelaScan — Melanoma detection in a snap!

Melanoma, the deadliest type of skin cancer, is actually easily treatable if detected at an early stage. However, due to how easy it is to overlook on the skin and how difficult it is to recognize by non-specialized doctors, it often goes undetected until it's too late. So we built a decision support app to help doctors in the **diagnosis of melanoma in a very easy to use and rapid fashion**, <20 seconds from app launch to analysis result.

<a href="https://youtu.be/0DBCGSW64r4"><img width="50%" src="https://joongwonseo.github.io/projects/melascan/home_cropped.gif" alt="Demo Video" /></a> <br/>*(Click the image to see a short demo on YouTube)*

The MelaScan app takes a photo of the mole on the patient's skin. With the help of our AI model, it instantly returns a probability of this mole being a case of melanoma.

![](https://d112y698adiu2z.cloudfront.net/photos/production/software_photos/002/240/178/datas/gallery.jpg)


## Tech Stack
Cross-platform app: Flutter

REST API local server: Flask on Python

Image analysis: PyTorch, using Ensemble of CNNs (EfficientNets)


## Limitations
One of the biggest challenges we ran into was the **extremely imbalanced dataset**. Only 1.76% of the whole dataset contains positive examples (melanoma). Another problem also related to the data is the lack of variety, especially little data on people of color, potentially leading to poor performance and **bias against minorities**.
More metadata in the future could increase accuracy: e.g. exposure for UV-Radiation, usage of skin care products, a marker for the size of the mole, hereditary and nodule mole type.

We hope that the accessibility of this app will allow it to be deployed world-wide, greatly aiding in gathering more real data from especially less represented parts of the world and creating a much more balanced and racially diverse dataset, thus contributing to the medical research effort and improving the AI detection.


## Getting Started

### App
The Flutter app should be installed on the smartphone and be used to take pictures, which will be sent to the server to be analysed. It can be installed onto device as normal (see Flutter documentation). In the settings menu, you can set the IP and port of the analysis server (see below).

### Server
The python server should run on a server computer in the clinic network. While any modern computer should work with CPU inference, the analysis time might suffer greatly if not using a GPU. You can reduce the size of the ensemble to reduce the analysis time. But on any mid-tier NVIDIA GPU, the full-size model should not take more than 3-5 seconds per analysis (around 3s on RTX3060).

Clone the repository, download the pretrained models from the link in `server/model/readme.txt`, install the python requirements: `cd server && pip install -r requirements.txt` then simply run the server: `python server.py`

Once the server is running, Flask will print the local IP and port of the server, which can be then set in the app settings. This allows for local hosting of the server for medical privacy, or potentially for a centrally hosted server in the cloud (with doctor & patient consent).

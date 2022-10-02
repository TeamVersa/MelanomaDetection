from flask import Flask, jsonify, request
import time


from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras import preprocessing
#from efficientnet.tfkeras import EfficientNetB4
import efficientnet.tfkeras as efn

EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6]

EFF_NETS = [6,6,6,6,6]

def build_model(dim=128, ef=0):
    inp = tf.keras.layers.Input(shape=(dim,dim,3))
    base = EFNS[ef](input_shape=(dim,dim,3),weights='imagenet',include_top=False)
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp,outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 
    model.compile(optimizer=opt,loss=loss,metrics=['AUC'])
    return model


print('building model...')
model = build_model(dim=384, ef=EFF_NETS[0])


print('loading weights...')
model.load_weights("chris-fold-0.h5")

# classifier_model = "fold-0.h5"  # MODEL PATH
# model = load_model(classifier_model)
print(f'loaded model = {model.summary()}')


def predict(image, model):
    a, b = image.size
    width, height = min(a, b), max(a, b)
    dimension = width
    padding = (height - width) / 2
    test_image = image.crop((padding, 0, height-padding, width))
    test_image.save('image_cropped.jpg', "JPEG")
    test_image = test_image.resize((384, 384))
    test_image.save('image_resized.jpg')
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = {0: 'Benign', 1: 'Malignant'}
    predictions = model.predict(test_image)
    # scores = tf.nn.sigmoid(predictions[0])
    scores = predictions[0]
    #print(f'prediction scores = {scores}')
    result = {
        'prediction': 'Benign' if scores < 0.5 else 'Malignant',
        'confidence': (100 * np.average(scores)).round(2)
    }

    return result


app = Flask(__name__)


@app.route('/melanoma', methods=['POST'])
def get_tasks():
    i = request.files.get('image', '')
    print(f'received image {i}')
    i.save('image.jpg')
    # time.sleep(2.0)

    img_path = 'image.jpg'
    with Image.open(img_path) as img:
        result = predict(img, model)
        print(result)

    return jsonify(result)


if __name__ == '__main__':
    print('hello world!')
    app.run(debug=True, host='0.0.0.0')

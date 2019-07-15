
from tensorflow.python.keras.models import load_model
import tensorflow as tf

import numpy as np
from PIL import Image
import sys
import os

MODEL_NAME = 'flowers.hd5'
dict = {
    0: 'maine_coon_cat',
    1: 'singapura_cat',
    2: 'ocelot_cat',
    3: 'turkish_van_cat',
}
graph = tf.get_default_graph()


def classify(model, image):
    global graph
    with graph.as_default():
        result = model.predict(image)
        themax = np.argmax(result)

    return dict[themax], result[0][themax], themax


def load_image(image_fname):
    img = Image.open(image_fname).resize((249, 249))
    imgarray = np.array(img)/255.0
    final = np.expand_dims(imgarray, axis=0)
    return final


def main():
    if len(sys.argv) != 3:
        print("does't have enough arguments, cat_predict <img_folder> <target>")
        exit(-1)
    test_folder = sys.argv[1]
    target = sys.argv[2]
    model = load_model(MODEL_NAME)
    correct_num = 0.
    total = 0
    for img_path in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_path)
        img = load_image(img_path)
        label, prob, _ = classify(model, img)
        print("We think with certainty %3.2f that it is %s." % (prob, label))
        if target[0] == label[0]:
            correct_num += 1
        total += 1

    print("final accuracy of %s is %3.f" % (target, correct_num/total))


if __name__ == '__main__':
    main()

from tensorflow.python.keras.models import load_model
import tensorflow as tf

import numpy as np
from PIL import Image
import sys
import os

MODEL_NAME = 'cat_net.hd5'
dict = {
    0: 'maine_coon_cat',
    1: 'ocelot_cat',
    2: 'singapura_cat',
    3: 'turkish_van_cat',
}
graph = None


def classify(model, image):
    global graph
    if graph is None:
        graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        result = model.predict(image)
        themax = np.argmax(result)

    return dict[themax], result[0][themax], themax


def load_image(image_fname):
    img = Image.open(image_fname).resize((256, 256))
    imgarray = np.array(img) / 255.0
    final = np.expand_dims(imgarray, axis=0)
    return final


class CatModel:

    def __init__(self, model_name):
        self.model = load_model(model_name)

    def predict(self, img: Image):
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        final_img = np.expand_dims(img_array, axis=0)
        label, prob, _ = classify(self.model, final_img)
        return label, prob


def test_folder_acc(model, test_folder, target):
    correct_num = 0.
    total = 0
    for img_path in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_path)
        img = load_image(img_path)
        try:
            label, prob, _ = classify(model, img)
            print(img_path, "We think with certainty %3.2f that it is %s." % (prob, label))
            if target[0] == label[0]:
                correct_num += 1
            total += 1
        except:
            print("this img has some problem")

    print("final accuracy of %s is %3.2f" % (target, correct_num / total))


def main(multi_type=False):
    if len(sys.argv) != 4:
        print("does't have enough arguments, gesture_predict <model name> <img_folder> <target>")
        exit(-1)
    MODEL_NAME = sys.argv[1]
    test_folder = sys.argv[2]
    target = sys.argv[3]
    model = load_model(MODEL_NAME)
    if multi_type:
        for one_test_folder in os.listdir(test_folder):
            one_cat_folder = os.path.join(test_folder, one_test_folder)
            test_folder_acc(model, one_cat_folder, one_test_folder)
    else:
        test_folder_acc(model, test_folder, target)


if __name__ == '__main__':
    main(multi_type=False)

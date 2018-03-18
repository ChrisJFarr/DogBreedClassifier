import cv2
import numpy as np
from keras.models import load_model
from keras.applications import mobilenet
from glob import glob


class DogBreedClassifier:

    def __init__(self):
        # load start net (the model runs at start of program)
        self.start_model = load_model('start_net.h5', custom_objects={
            'relu6': mobilenet.relu6,
            'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
        # load saved trained model
        self.model = load_model('mobilenet.h5', custom_objects={
            'relu6': mobilenet.relu6,
            'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
        # load dog classes
        self.breeds = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
        # load pre_classify classes for dog, cat, human, neither
        self.classes = [name[21:-1] for name in sorted(glob("custom_images/images/*/"))]

    def run(self, image_path: str):
        """
        First check to see what type of object is in image.
        If human, dog, or cat detected the predict dog breed.
        Print messages.
        :param image_path: String
        :return: None
        """
        img = self.get_image(image_path)
        preclass, proceed = self.pre_classify(img)
        print(preclass)
        if proceed:
            pred_dict = self.predict(img)
            print(self.decide_breed(pred_dict))
        return None

    @staticmethod
    def get_image(image_path: str):
        img = cv2.imread(image_path)
        assert img is not None, "Check image path, no image found."
        img = img / 1.  # Convert to float to avoid error, keras bug
        return img

    def pre_classify(self, img) -> (str, bool):
        """
        Take location of image and return if it's a human, cat, dog, or neither.
        Example output:
        This looks like a human, I'll pretend it's a dog.
        This looks like a cat, I'll pretend it's a dog.
        I don't see a dog, cat, or a human in this photo... are you trying to trick me?
        :param img:
        :return:
        """
        # Convert image to 224 * 224
        img = cv2.resize(img, dsize=(224, 224))
        # Preprocess data
        img = np.expand_dims(img, axis=0)
        # Preprocess input for mobilenet
        img = mobilenet.preprocess_input(img)
        # generate predictions
        prediction = self.start_model.predict(img)
        # Class
        pred_class = self.classes[np.argmax(prediction)]
        # Response dictionary
        response_dict = dict([("dog", ("This looks like a dog, let me guess the breed.", True)),
                              ("cat", ("This looks like a cat, I'll pretend it's a dog.", True)),
                              ("person", ("This looks like a person, I'll pretend it's a dog.", True)),
                              ("other", ("I don't see a dog, cat, or a human in this photo... "
                                         "are you trying to trick me?", False))])
        return response_dict[pred_class]

    def predict(self, img) -> dict:
        """
        Generate prediction dictionary with top 3 predicted dog breeds
        :param img:
        :return:
        """
        # Convert image to 224 * 224
        img = cv2.resize(img, dsize=(224, 224))
        # Preprocess data
        img = np.expand_dims(img, axis=0)
        # Preprocess input for mobilenet
        img = mobilenet.preprocess_input(img)
        # generate top 3 predictions
        prediction = self.model.predict(img)[0]
        # print(np.argmax(prediction))
        top_3 = prediction.argsort()[-3:][::-1]
        top_3_pred_values = [prediction[i] for i in top_3]
        # List top 3 breeds in order
        prediction_dict = dict(zip([self.breeds[i] for i in top_3], top_3_pred_values))

        return prediction_dict

    @staticmethod
    def decide_breed(pred: dict) -> str:
        """
        Allow for mixed breed predictions.
        if 2nd highest > .25 and 1st highest less than .6 then mixed breed (2)
        if top 2 < .6 combined and top 3 > .75 then 3 breed mix
        # I think that is a Yorkie purebred
        # I think that is a Yorkie and Japanese Chin mix
        # I think that is a Yorkie, Japanese Chin, and Lab mix
        :param pred: Prediction dict from predict method
        :return: String to display as prediction output
        """
        for k, v in pred.items():
            pred[k.replace("_", " ")] = pred.pop(k)

        top1 = sorted(pred, key=lambda k: pred[k])[-1]
        top2 = sorted(pred, key=lambda k: pred[k])[-2]
        top3 = sorted(pred, key=lambda k: pred[k])[-3]

        if pred[top1] >= .6:
            breed = "I think this is a %s purebred." % top1
        elif (pred[top2] > .25) & (pred[top1] < .6):
            breed = "I think this is a %s and %s mix." % (top1, top2)
        elif (pred[top1] + pred[top2] < .6) & (sum(pred.values()) > .75):
            breed = "I think this is a %s, %s, and %s mix." % (top1, top2, top3)
        else:
            breed = "I'm honestly not that sure. \n" \
                    "Could be a %s, could be a %s, could be a %s. \n" \
                    "Or maybe it's a mix of all those and something else." \
                    % (top1, top2, top3)
        return breed

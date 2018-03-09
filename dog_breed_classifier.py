# from keras.models import load_model
# from keras.applications import ResNet50
import cv2
# from glob import glob
import numpy as np
# sys.path = [os.path.dirname(os.getcwd())] + sys.path


class DogBreedClassifier:

    def __init__(self):
        # TODO load saved trained model
        # self.model = load_model("../best_dog_breed_model")
        # TODO load resnet convolutional layers
        # self.resnet_conv = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        # TODO load dog classes
        # self.breeds = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
        pass

    def run(self, image_path: str) -> str:
        img = self.get_image(image_path)
        pred_dict = self.predict(img)

        return "Done"

    @staticmethod
    def get_image(image_path: str):
        img = cv2.imread(image_path)
        assert img is not None, "Check image path, no image found."
        return img

    def predict(self, img) -> dict:
        # TODO: Write a function that takes a path to an image as input
        # and returns the dog breed that is predicted by the model.
        # TODO take path and load image, handle any type?
        # TODO try, if None then wrong image address
        # TODO convert image to 224 * 224
        img = cv2.resize(img, dsize=(224, 224))
        # TODO convert input to resnet shape and transform using convolutional layers
        conv_data = self.resnet_conv.predict(np.expand_dims(img, axis=0))
        # TODO generate top 3 predictions
        prediction = self.model.predict(conv_data)[0]
        # print(np.argmax(prediction))
        top_3 = prediction.argsort()[-3:][::-1]
        top_3_pred_values = [prediction[i] for i in top_3]

        # List top 3 breeds in order
        prediction_dict = dict(zip([self.breeds[i] for i in top_3], top_3_pred_values))
        return prediction_dict

    @staticmethod
    def decide_breed(pred: dict) -> str:
        # TODO show mixed breed predictions
        # TODO if 2nd highest > .25 and 1st highest less than .6 then mixed breed (2)
        # TODO if top 2 < .6 combined and top 3 > .75 then 3 breed mix
        # I think that is a Yorkie purebred
        # I think that is a Yorkie and Japanese Chin mix
        # I think that is a Yorkie, Japanese Chin, and Lab mix
        # Store keys
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
                    "Or maybe it's a mix of all those and something else."\
                    % (top1, top2, top3)
        return breed

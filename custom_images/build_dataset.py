import os
import numpy as np
from urllib import request
import cv2
import scipy.misc
# Download urls from ImageNet
# http://image-net.org/download-imageurls
# Download bounding box annotations for desired classes
# http://image-net.org/api/text/imagenet.bbox.obtain_synset_wordlist

# Create list of annotation file names


# Assume every folder in custom image directory is
# dir_list = os.listdir("custom_images")


class ImageNetData:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def build_annotations(self, dir_list) -> set:
        # Create list of image id's based on folders containing wordnet annotations
        annotations_list = []
        for directory in dir_list:
            try:
                file_list = os.listdir(os.path.join(self.base_dir, "annotations", directory))
                file_list = [f.split(".")[0] for f in file_list]
                annotations_list += file_list
            except NotADirectoryError:
                pass
        annotations_set = set(annotations_list)

        return annotations_set

    def build_urls(self, url_txt: str, annotations_set: set) -> list:

        # Download images in list
        # Read in all urls
        url_txt_file = os.path.join(self.base_dir, url_txt)
        url_list_raw = open(url_txt_file, errors="ignore").readlines()

        # Loop through annotations list to filter to urls
        url_list = []
        for url in url_list_raw:
            wordnet_id, _, url = url.partition("\t")
            url = url.strip("\n")
            if wordnet_id in annotations_set:
                url_list.append((url, wordnet_id))

        return url_list

    def save_images(self, url_list: list, destination: str) -> None:
        # destination is the name of the folder in
        fail = 0
        succeed = 0
        for url, wordnet_id in url_list:
            try:
                img = self.url_to_image(url)
                save_file = os.path.join(self.base_dir, "images", destination, wordnet_id + ".jpg")
                scipy.misc.toimage(img, cmin=0.0, cmax=...).save(save_file)
                succeed += 1
            except Exception as e:
                print(e)
                print("for %s" % wordnet_id)
                fail += 1
                pass
        print("%s files were saved. %s failed to load." % (succeed, fail))
        return None

    @staticmethod
    def url_to_image(url):
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp = request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # return the image
        return image


# Cats completed: 352 examples
# imdata = ImageNetData("custom_images")
# anno_set = imdata.build_annotations(["cat"])
# urls = imdata.build_urls("fall11_urls.txt", anno_set)
# imdata.save_images(urls, "cat")
# cat_files = os.listdir("custom_images/images/cat")

# imdata = ImageNetData("custom_images")
# anno_set = imdata.build_annotations(["bathroom", "bedroom", "door", "fruit",
#                                      "hamster", "house", "kitchen", "lamp",
#                                      "table", "television_room"])
# urls = imdata.build_urls("fall11_urls.txt", anno_set)
# imdata.save_images(urls, "other")

# imdata = ImageNetData("custom_images")
# anno_set = imdata.build_annotations(["dog"])
# urls = imdata.build_urls("fall11_urls.txt", anno_set)
# imdata.save_images(urls, "dog")

imdata = ImageNetData("custom_images")
anno_set = imdata.build_annotations(["people"])
urls = imdata.build_urls("fall11_urls.txt", anno_set)
imdata.save_images(urls, "person")

# TODO start here, build remaining image datasets
# Set destination folder
# TODO write algorithm to populate an image folder based on a list of directories with annotations
# TODO create a reusable class

os.path.join("custom_images\\images\\cat", "123123" + ".jpg")

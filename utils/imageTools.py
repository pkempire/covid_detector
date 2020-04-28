from keras.preprocessing.image import img_to_array

##Util from Pyimagesearch
class ImageToArrayPreprocessor:
    def __init__(self, data_format=None):
        self.dataFormat = data_format

    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)

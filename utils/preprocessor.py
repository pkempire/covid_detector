import cv2
##Util from Pyimagesearch
class preprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, input_image):

        return cv2.resize(input_image, (self.width, self.height),
                              interpolation=self.inter)



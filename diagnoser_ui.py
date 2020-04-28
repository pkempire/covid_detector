import cv2
import numpy as np
from keras.models import load_model
from tkinter import Frame, Tk, BOTH, Label, filedialog, Text
from PIL import Image, ImageTk
from tkinter.ttk import *
import os

screenWidth = "1050"
screenHeight = "780"
windowTitle = "Covid 19 Detector "
imgDims = 480

model = load_model('models/model_keras_215.hdf5')

def getPrediction (imagePath):
    test_data = []
    img = cv2.imread(imagePath, 0)
    img = cv2.resize(img, (50, 50))
    img = np.dstack([img, img, img])
    img = img.astype('float32') / 255
    test_data.append(img)
    prediction = model.predict(np.array(test_data))
    _prediction = round(prediction[0][0]*100, 3)
    if _prediction > 50:
        _prediction = "                                                       Covid 19 detected"
    elif _prediction < 50:
        _prediction = "                                                       Normal lungs detected"
    intPred = round(prediction[0][0], 2)
    outputContent = _prediction + "\n\n"+"                                                   Neural Network Output : " + str(intPred)+ "\n\n                                   The closer to 1 the higher the chance that Covid 19 is present"
    return outputContent

class Window(Frame):
    result = ""
    resultField = None
    lastImg = None

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)
        back = Image.open(r"C:\Users\Parth\Music\Music\plain-white-background.jpg")
        render = ImageTk.PhotoImage(back)
        img = Label(self, image=render)
        img.image = render
        img.place(x=(int(screenWidth)/2)-back.width/2, y=((int(screenHeight)/2))-back.height/2-80)
        self.lastImg = img
        self.resultField = Text(self,  width=int(screenWidth), height=13)
        self.resultField.pack()
        self.resultField.place(x=0, y=int(screenHeight)-200)

    def addResult (self, value):
        self.resultField.delete("1.0","end")
        self.result = ""
        self.resultField.insert(1.0, value)

root = Tk()
main = Window(root)
root.wm_title(windowTitle)
root.geometry(screenWidth + "x" + screenHeight)

root.iconbitmap("covid_bm8_icon.ico")



def getImage():
    currdir = os.getcwd()
    image_file = filedialog.askopenfile(mode ='r', parent=root, initialdir=currdir)
    root.wm_title(windowTitle)
    loadImage(image_file.name)

def loadImage(filename):
    main.lastImg.destroy()
    load = Image.open(filename)
    load = load.resize((imgDims, imgDims))
    render = ImageTk.PhotoImage(load)
    img = Label(image=render)
    img.image = render
    img.place(x=(int(screenWidth)/2)-imgDims/2, y=((int(screenHeight)/2))-imgDims/2-80)
    main.result +=  "                                                             Result\n                                         " + filename+"\n\n"
    main.result += getPrediction (filename)
    print(main.result)
    main.lastImg = img
    main.addResult(main.result)

btn = Button(root, text = 'Click to select image', command = getImage)
btn.place(x=425,y=0, width=180, height=60 )

root.mainloop()

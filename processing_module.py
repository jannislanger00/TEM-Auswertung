import sys
import numpy as np
#from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel, QPushButton, QLineEdit
#from PyQt5.QtCore import Qt
import cv2
#from Interface import ParticleUserinterface
PATH = r'Results/gui_img.jpg'  # Path jpg File
DPI = 100
SCALE = 0.5


class ParticleProcessing():

    def __init__(self, path):
        # init parameters
        self.text_input = False
        self.peaksTresh = 0.45
        self.blockSize = 41
        self.C = 3
        self.blur1 = 13
        BGthresh = 128

        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.img = cv2.imread(path)
        self.gray = self.grayBlur()
        self.BGimg = self.bg_thresh(BGthresh)
        # Template
        self.particle_size = 20  # pixel
        self.gap = 14
        # start up


    def grayBlur(self):
        return cv2.GaussianBlur(cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY), (self.blur1, self.blur1), 0)

    def bg_thresh(self, thresh):
        th, bw = cv2.threshold(self.gray, thresh, 255, cv2.THRESH_BINARY_INV)
        morf = cv2.morphologyEx(bw, cv2.MORPH_OPEN, self.kernel5)
        self.BGimg = bw
        print(self.BGimg.shape)
        res = cv2.cvtColor(morf, cv2.COLOR_GRAY2BGR)

        #cv2.imshow('res', res)
        return res

    def ad_thresh(self, block):
        self.blockSize = block
        #self.C = c
        gray = cv2.GaussianBlur(cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY), (self.blur1, self.blur1), 0)
        ad_th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                      self.blockSize, self.C)
        #self.BGimg = cv2.cvtColor(self.BGimg, cv2.COLOR_BGR2GRAY)
        #print(self.BGimg.shape)
        #masked = cv2.bitwise_and(ad_th, ad_th, mask=self.BGimg)
        #morf = cv2.morphologyEx(masked, cv2.MORPH_OPEN, self.kernel3)
        #res = cv2.cvtColor(self.BGimg, cv2.COLOR_GRAY2BGR)
        #cv2.imshow('img', self.BGimg)

        #cv2.imshow('res', res)
        return ad_th

    def thresh(self, val):
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        th, bw = cv2.threshold(self.img, val, 255, cv2.THRESH_BINARY_INV)
        gray2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return gray2

    def hi(self, x):
        print(x)

    def stackImages(self, scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                    None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            hor_con = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,
                                             scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            ver = hor
        return ver

    def resizeImg(self, img, scale):
        res = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        return res

if __name__ == "__main__":
    a = ParticleProcessing(PATH)
    a.ad_thresh(51)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
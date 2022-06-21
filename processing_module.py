import sys
import numpy as np
#from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel, QPushButton, QLineEdit
#from PyQt5.QtCore import Qt
import cv2


class ParticleDetection():

    def __init__(self):
        # init Values
        self.bg_thresh = 0.35
        self.peaksTresh = 0.45
        self.blockSize = 41
        self.C = 2
        # Template
        self.particle_size = 20  # pixel
        self.gap = 14
        # start up
        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.text_input = False

    def bg_change(self, img, val):
        self.bg_thresh = val
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (13, 13), 0)
        th, bw = cv2.threshold(gray, self.bg_thresh * np.amax(gray), 255, cv2.THRESH_BINARY_INV)
        res = cv2.morphologyEx(bw, cv2.MORPH_OPEN, self.kernel5)
#        if self.comp_button.isChecked():
#            res = self.img.copy()
#            contours, hierarchy = cv2.findContours(self.output_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#            cv2.drawContours(res, contours, -1, (0, 255, 0), 2)
#        else:
#            res = self.output_bw
        return res

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
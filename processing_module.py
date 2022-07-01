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
        self.bgThresh = 128
        self.peaksThresh = 0.45
        self.blockSize = 41
        self.C = 3
        self.blur1 = 13
        BGthresh = 128
        # Template
        self.particle_size = 20  # pixel
        self.gap = 14
        self.DistTempl()
        #init images and kernels
        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.img = cv2.imread(path)
        self.gray = self.grayBlur()
        self.bgMask = self.bg_thresh(BGthresh)
        _ = self.ad_thresh(5)

        # start up

    def DistTempl(self):
        borderSize = int((self.particle_size / 2) + self.gap)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (2 * (borderSize - self.gap) + 1, 2 * (borderSize - self.gap) + 1))
        kernel2 = cv2.copyMakeBorder(kernel2, self.gap, self.gap, self.gap, self.gap,
                                     cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
        self.distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        #return distTempl

    def grayBlur(self):
        return cv2.GaussianBlur(cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY), (self.blur1, self.blur1), 0)

    def bg_thresh(self, thresh=None):
        if thresh:
            self.bgThresh = thresh
        th, bw = cv2.threshold(self.gray, self.bgThresh, 255, cv2.THRESH_BINARY_INV)
        self.bgMask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, self.kernel5)
        res = self.bgMask
        #cv2.imshow('res', res)
        return res

    def ad_thresh(self, block = None, c = None):
        if block:
            self.blockSize = block
        if c:
            self.C = c
        ad_th = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                      self.blockSize, self.C)
        #self.BGimg = cv2.cvtColor(self.BGimg, cv2.COLOR_BGR2GRAY)
        #print(self.BGimg.shape)
        masked = cv2.bitwise_and(ad_th, ad_th, mask=self.bgMask)
        self.ad_out = cv2.morphologyEx(masked, cv2.MORPH_OPEN, self.kernel3)
        res = self.ad_out
        return res

    def p_thresh(self, thresh=None):
        if thresh:
            self.peaksThresh = thresh
        dist = cv2.distanceTransform(self.ad_out, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        borderSize = int((self.particle_size / 2) + self.gap)
        distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize,
                                        cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
        self.nxcor = cv2.matchTemplate(distborder, self.distTempl, cv2.TM_CCOEFF_NORMED)
        # Peaks
        mn, mx, _, _ = cv2.minMaxLoc(self.nxcor)
        th, peaks = cv2.threshold(self.nxcor, mx * self.peaksThresh, 255, cv2.THRESH_BINARY)
        peaks8u = cv2.convertScaleAbs(peaks)
        self.peaks8u = cv2.convertScaleAbs(peaks)
        res = self.peaks8u
        return res

    def hi(self, x):
        print(x)

    def process_all(self):
        self.bg_thresh()


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
    a.ad_thresh()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
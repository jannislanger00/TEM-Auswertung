import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

#from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel, QPushButton, QLineEdit
#from PyQt5.QtCore import Qt
PATH = r'Results/gui_img.jpg'  # Path jpg File
DPI = 100
SCALE = 0.5


class ParticleProcessing():

    def __init__(self, path):
        # init parameters
        self.text_input = False
        self.bgThresh = 200
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
        self.process_all()

        # start up

    def DistTempl(self, size=None):
        if size is not None:
            self.particle_size = size
        borderSize = int((self.particle_size / 2) + self.gap)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (2 * (borderSize - self.gap) + 1, 2 * (borderSize - self.gap) + 1))
        kernel2 = cv2.copyMakeBorder(kernel2, self.gap, self.gap, self.gap, self.gap,
                                     cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
        self.template = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return self.template

    def grayBlur(self):
        return cv2.GaussianBlur(cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY), (self.blur1, self.blur1), 0)

    def bg_thresh(self, thresh=None, contours = False):
        if thresh is not None:
            self.bgThresh = thresh
        # Background Processing
        th, bw = cv2.threshold(self.gray, self.bgThresh, 255, cv2.THRESH_BINARY_INV)
        self.bgMask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, self.kernel5)
        # Prepare Result
        if contours:
            res = self.img.copy()
            contours, hierarchy = cv2.findContours(self.bgMask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res, contours, -1, (0, 255, 0), 2)
        else:
            res = self.bgMask
        return res

    def ad_thresh(self, block=None, c=None, contours=False):
        if block is not None:
            self.blockSize = block
        if c is not None:
            self.C = c
        # Adaptive Processing
        ad_th = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                      self.blockSize, self.C)
        masked = cv2.bitwise_and(ad_th, ad_th, mask=self.bgMask)
        self.ad_out = cv2.morphologyEx(masked, cv2.MORPH_OPEN, self.kernel3)
        # Prepare Result
        if contours:
            res = self.img.copy()
            contours, hierarchy = cv2.findContours(self.ad_out, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res, contours, -1, (0, 255, 0), 2)
        else:
            res = self.ad_out
        return res

    def p_thresh(self, thresh=None, contours = False):
        if thresh is not None:
            self.peaksThresh = thresh
        # Distance Transform ~ Match template
        self.dist = cv2.distanceTransform(self.ad_out, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        borderSize = int((self.particle_size / 2) + self.gap)
        distborder = cv2.copyMakeBorder(self.dist, borderSize, borderSize, borderSize, borderSize,
                                        cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
        self.nxcor = cv2.matchTemplate(distborder, self.template, cv2.TM_CCOEFF_NORMED)
        # Get Peaks
        mn, mx, _, _ = cv2.minMaxLoc(self.nxcor)
        th, peaks = cv2.threshold(self.nxcor, mx * self.peaksThresh, 255, cv2.THRESH_BINARY)
        peaks8u = cv2.convertScaleAbs(peaks)
        self.peaks8u = cv2.convertScaleAbs(peaks)
        # Prepare Result
        if contours:
            res = self.img.copy()
            contours, hierarchy = cv2.findContours(self.peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res, contours, -1, (0, 255, 0), 2)
        else:
            res = self.peaks8u
        return res

    def hi(self, x):
        print(x)

    def process_all(self):
        self.bg_thresh()
        self.ad_thresh()
        res = self.p_thresh()
        return res

    def calcParticles(self):
        self.process_all()
        self.img_draw = self.img.copy()
        contours, hierarchy = cv2.findContours(self.peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        counter = 0
        size = np.zeros(len(contours))
        for i in range(len(contours)):
            # cv2.drawContours(self.img_draw, contours, i, (0, 0, 255), 1)
            x, y, w, h = cv2.boundingRect(contours[i])
            _, mx, _, mxloc = cv2.minMaxLoc(self.dist[y:y + h, x:x + w])
            if 7.5 <= mx <= 400:
                center = (int(mxloc[0] + x), int(mxloc[1] + y))
                radius = int(np.round(mx))
                cv2.circle(self.img_draw, center, radius, (255, 0, 0), 1)
                size[counter] = mx * 2
                counter += 1

        size = np.trim_zeros(size)
        mean = np.sum(size) / counter
        cv2.putText(self.img_draw, "counter: " + str(int(counter)), (75, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(self.img_draw,
                    "av. size: : " + str(round(mean, 2)),
                    (75, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #res = resizeImg(self.img_draw, SCALE)
        cv2.imshow('img', self.img_draw)
        ### Histogramm
        size.tofile('Files/sample.csv', sep='\n')
        file = open("Files/sample.csv")
        numpy_array = np.array(size)
        n, bins, patches = plt.hist(x=numpy_array, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Size')
        plt.ylabel('Frequency')
        plt.title('Histogram')
        #plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.show()

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
    res = a.DistTempl(15)
    #res = a.calcParticles()
    cv2.imshow('eres', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
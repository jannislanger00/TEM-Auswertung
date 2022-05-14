import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel, QPushButton, QLineEdit
from PyQt5.QtCore import Qt
import cv2
from Aufgabe_Catalyst import read_dm3_file, dm3_to_cv
from gui import Export_img

PATH = r'Results/gui_img.jpg'  # Path jpg File
DPI = 100
SCALE = 0.8


def import_img(path):
    img_raw = cv2.imread(path)
    return img_raw


def resizeImg(img, scale):
    res = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    return res


def stackImages(scale, imgArray):
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


class ParticleDetection(QMainWindow):

    def __init__(self, path):
        super().__init__()
        # start Values
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
        self.img = import_img(path)
        self.text_input = False
        self.update_all()
        # Toolbar
        self.setGeometry(50, 50, 750, 900)
        self.setWindowTitle("Particle Detection")
        self.createUI()
        self.show()

    def createUI(self):
        # Create textbox
        self.textbox = QLineEdit(self)
        self.textbox.move(300, 20)
        self.textbox.resize(80, 40)
        self.textbox.setText(str(self.particle_size))
        if self.textbox.text().isnumeric() == True:
            self.text_input = True
        elif self.textbox.text().isnumeric() == False:
            self.text_input = False
            self.textbox.setText('error')
        label_partSize = QLabel('Particle Size (px)', self)
        label_partSize.setGeometry(30, 20, 270, 60)

        # Background Thresholt
        bg_slider = QSlider(Qt.Horizontal, self)
        bg_slider.setRange(0, 1 * DPI)
        bg_slider.setGeometry(300, 100, 300, 60)
        bg_slider.setSliderPosition(int(self.bg_thresh * DPI))
        self.label_bg_thresh = QLabel('Background Thresholt', self)
        self.label_bg_thresh.setGeometry(30, 100, 270, 60)
        self.label_bg_thresh.setText('Background Thresholt: ' + str(self.bg_thresh))
        bg_slider.valueChanged[int].connect(self.changeBgThresh)

        # Peaks
        peaks_slider = QSlider(Qt.Horizontal, self)
        peaks_slider.setRange(0, 1 * DPI)
        peaks_slider.setGeometry(300, 300, 300, 60)
        peaks_slider.setSliderPosition(int(self.peaksTresh * DPI))
        self.label_peaks = QLabel('Peaks Thresh', self)
        self.label_peaks.setGeometry(30, 300, 270, 60)
        self.label_peaks.setText('Peaks Thresh: ' + str(self.peaksTresh))
        peaks_slider.valueChanged[int].connect(self.changePeaksThresh)

        # Blocksize
        blockSize_slider = QSlider(Qt.Horizontal, self)
        blockSize_slider.setRange(1, 1 * DPI // 2)
        blockSize_slider.setGeometry(300, 200, 300, 60)
        blockSize_slider.setSliderPosition(int(self.blockSize // 2))
        self.label_blockSize = QLabel('(Adaptive Thresh) Block Size', self)
        self.label_blockSize.setGeometry(30, 200, 270, 60)
        self.label_blockSize.setText('Block Size: ' + str(self.blockSize))
        blockSize_slider.valueChanged[int].connect(self.changeBlockSize)
        # Mean C
        self.textbox_C = QLineEdit(self)
        self.textbox_C.move(680, 200)
        self.textbox_C.resize(70, 40)
        self.textbox_C.setText(str(self.C))
        self.textbox_C.textChanged[str].connect(self.textCupdate)
        label_C = QLabel('C:', self)
        label_C.setGeometry(630, 200, 40, 40)
        # Update Button
        update_button = QPushButton('Update', self)
        update_button.resize(200, 32)
        update_button.move(150, 500)
        update_button.clicked.connect(self.update_all)

        res_button = QPushButton('Aprox. Particles', self)
        res_button.resize(200, 32)
        res_button.move(350, 500)
        res_button.clicked.connect(self.calcParticles)

        self.comp_button = QPushButton('Compare', self)
        self.comp_button.setCheckable(True)  # Switch
        self.comp_button.resize(200, 32)
        self.comp_button.move(150, 700)
        self.comp_button.clicked.connect(self.compareImg)

        self.original_button = QPushButton('show Image', self)
        self.original_button.resize(200, 32)
        self.original_button.move(350, 700)
        self.original_button.clicked.connect(self.showOriginal)

    def textCupdate(self):
        if self.textbox_C.text().isnumeric() == True:
            self.C = int(self.textbox_C.text())
        elif self.textbox_C.text().isnumeric() == False:
            self.textbox_C.setText('error')
            self.textbox_C.selectAll()

        self.updateAdThresh()

    def changeBgThresh(self, value):
        self.bg_thresh = value / DPI
        self.k = 0
        self.updateBgTresh()
        self.label_bg_thresh.setText('Background Thresholt: ' + str(self.bg_thresh))

    def changePeaksThresh(self, value):
        self.peaksTresh = value / DPI
        self.k = 3
        self.label_peaks.setText('Peaks Thresh: ' + str(self.peaksTresh))
        self.updatePeaks()

    def changeBlockSize(self, value):
        self.blockSize = value * 2 + 1
        self.k = 1
        self.label_blockSize.setText('Block Size: ' + str(self.blockSize))
        self.updateAdThresh()

    def updateBgTresh(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (13, 13), 0)
        th, bw = cv2.threshold(gray, self.bg_thresh * np.amax(gray), 255, cv2.THRESH_BINARY_INV)
        self.output_bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, self.kernel5)
        if self.comp_button.isChecked():
            res = self.img.copy()
            contours, hierarchy = cv2.findContours(self.output_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res, contours, -1, (0, 255, 0), 2)
        else:
            res = self.output_bw
        res = resizeImg(res, SCALE)
        cv2.imshow('img', res)

    def updateAdThresh(self):
        ad_th = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                      self.blockSize, self.C)
        masked = cv2.bitwise_and(ad_th, ad_th, mask=self.output_bw)
        self.output_ad_th = cv2.morphologyEx(masked, cv2.MORPH_OPEN, self.kernel3)
        if self.comp_button.isChecked():
            res = self.img.copy()
            contours, hierarchy = cv2.findContours(self.output_ad_th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res, contours, -1, (0, 255, 0), 2)
        else:
            res = self.output_ad_th
        res = resizeImg(res, SCALE)
        cv2.imshow('img', res)

    def updatePeaks(self):
        mn, mx, _, _ = cv2.minMaxLoc(self.nxcor)
        th, peaks = cv2.threshold(self.nxcor, mx * self.peaksTresh, 255, cv2.THRESH_BINARY)
        self.peaks8u = cv2.convertScaleAbs(peaks)
        if self.comp_button.isChecked():
            res = self.img.copy()
            contours, hierarchy = cv2.findContours(self.peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res, contours, -1, (0, 0, 255), 2)
        else:
            res = self.peaks8u
        res = resizeImg(res, SCALE)
        cv2.imshow('img', res)

    def showOriginal(self):
        res = resizeImg(self.img, SCALE)
        cv2.imshow('img', res)

    def compare(self):
        pass

    def compareImg(self):
        img_ref = self.imgList[self.k]
        res = self.img.copy()
        if self.comp_button.isChecked() and img_ref is not None:
            contours, hierarchy = cv2.findContours(img_ref, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res, contours, -1, (0, 0, 255), 1)
        else:
            res = img_ref

        res = resizeImg(res, SCALE)
        cv2.imshow('img', res)

    def update_all(self):
        if self.text_input == True:
            self.particle_size = int(self.textbox.text())
        # bw
        gray1 = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray1, (13, 13), 0)
        th, bw = cv2.threshold(gray, self.bg_thresh * np.amax(gray), 255, cv2.THRESH_BINARY_INV)
        self.output_bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, self.kernel5)
        # Adthresh
        self.gray = cv2.GaussianBlur(gray, (31, 31), 0)
        ad_th = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                      self.blockSize, self.C)
        masked = cv2.bitwise_and(ad_th, ad_th, mask=self.output_bw)
        self.output_ad_th = cv2.morphologyEx(masked, cv2.MORPH_OPEN, self.kernel3)
        self.dist = cv2.distanceTransform(self.output_ad_th, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # nxcor
        borderSize = int((self.particle_size / 2) + self.gap)
        distborder = cv2.copyMakeBorder(self.dist, borderSize, borderSize, borderSize, borderSize,
                                        cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (2 * (borderSize - self.gap) + 1, 2 * (borderSize - self.gap) + 1))
        kernel2 = cv2.copyMakeBorder(kernel2, self.gap, self.gap, self.gap, self.gap,
                                     cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
        distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        self.nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
        # Peaks
        mn, mx, _, _ = cv2.minMaxLoc(self.nxcor)
        th, peaks = cv2.threshold(self.nxcor, mx * self.peaksTresh, 255, cv2.THRESH_BINARY)
        peaks8u = cv2.convertScaleAbs(peaks)
        self.peaks8u = cv2.convertScaleAbs(peaks)
        self.imgList = list((self.output_bw, self.output_ad_th, self.nxcor, self.peaks8u))
        self.imgArray = list(([self.output_bw, self.output_ad_th], [self.nxcor, self.peaks8u]))
        imgStack = stackImages(SCALE / 2, self.imgArray)
        cv2.imshow('img', imgStack)

    def calcParticles(self):
        self.update_all()
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
        cv2.putText(self.img_draw, "counter: " + str(int(counter)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2)
        cv2.putText(self.img_draw,
                    "av. size: : " + str(round(mean, 2)),
                    (75, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        res = resizeImg(self.img_draw, SCALE)
        cv2.imshow('img', res)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    #ex = Export_img('Bilder/2020_11_04-Pd-aAl__18.dm3')
    #cv2.waitKey(0)
    # path = ex.getImgPath()
    # print(path)
    fil = ParticleDetection(PATH)
    # cv2.destroyAllWindows()
    sys.exit(app.exec_())

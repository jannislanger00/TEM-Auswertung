from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QLabel, QPushButton, QLineEdit
from PyQt5.QtCore import Qt
import dm3_lib as dm3
import cv2
import sys
import numpy as np
from desginer_ui import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage
from processing_module import ParticleProcessing

PATH = r'Results/gui_img.jpg'  # Path jpg File
DPI = 100
SCALE = 0.8



class ParticleUserinterface(ParticleProcessing):

    def __init__(self, path):
        super().__init__(path)
        #self.path = path
        #self.image = cv2.imread(self.path)
        self.setupUi(MainWin)
        self.active = self.img
        self.show_image()
        self.show_temp()
        MainWin.show()

    def show_image(self, img=None):
        if img is not None:
            self.frame = img
        else:
            self.frame = self.active
        self.frame = QtGui.QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0],
                                  QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ImageWin.setPixmap(QtGui.QPixmap.fromImage(self.frame))

    def show_temp(self):
        tmpImg = self.template
        cv2.cvtColor(tmpImg, cv2.COLOR_GRAY2BGR)
        cv2.imshow(' e', tmpImg)
        tmpImg = QtGui.QImage(tmpImg.data, tmpImg.shape[1], tmpImg.shape[0],
                                  QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ParticleTemplate.setPixmap(QtGui.QPixmap.fromImage(tmpImg))

    def bg_change(self, val):
        self.bgThresh = val
        if self.ShowContours.isChecked():
            res = self.bg_thresh(contours=True)
        else:
            res = self.bg_thresh()
            res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

        self.show_image(res)

    def ad_change(self, val):
        self.blockSize = 2 * val + 1
        if self.ShowContours.isChecked():
            res = self.ad_thresh(contours=True)
        else:
            res = self.ad_thresh()
            res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        self.show_image(res)

    def p_change(self, val):
        self.peaksThresh = val / 100
        if self.ShowContours.isChecked():
            res = self.p_thresh(contours=True)
        else:
            res = self.p_thresh()
            res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        self.show_image(res)

    def mean_change(self, val):
        self.C = val
        if self.ShowContours.isChecked():
            res = self.ad_thresh(contours=True)
        else:
            res = self.ad_thresh()
            res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        self.show_image(res)

    def temp_change(self, val):
        self.particle_size = val
        self.DistTempl()

        self.show_temp()

    def update(self):
        self.process_all()
        self.show_image(self.img)

    def change_bool(self):
        pass

    #Generated script with QtDesigner
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(947, 1181)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.ImageWin = QtWidgets.QLabel(self.centralwidget)
        self.ImageWin.setGeometry(QtCore.QRect(10, 10, 921, 751))
        self.ImageWin.setAutoFillBackground(False)
        self.ImageWin.setText("")
        self.ImageWin.setPixmap(QtGui.QPixmap("../../../../Pictures/1500737818.jpg"))
        self.ImageWin.setScaledContents(True)
        self.ImageWin.setObjectName("ImageWin")

        self.BackgroundSlide = QtWidgets.QSlider(self.centralwidget)
        self.BackgroundSlide.setRange(0, 255)
        self.BackgroundSlide.setGeometry(QtCore.QRect(130, 831, 301, 21))
        self.BackgroundSlide.setOrientation(QtCore.Qt.Horizontal)
        self.BackgroundSlide.setObjectName("BackgroundSlide")
        self.BackgroundSlide.valueChanged[int].connect(self.bg_change)

        self.AdaptiveSlide = QtWidgets.QSlider(self.centralwidget)
        self.AdaptiveSlide.setRange(1, 75)
        self.AdaptiveSlide.setGeometry(QtCore.QRect(130, 880, 301, 21))
        self.AdaptiveSlide.setOrientation(QtCore.Qt.Horizontal)
        self.AdaptiveSlide.setObjectName("AdaptiveSlide")
        self.AdaptiveSlide.valueChanged[int].connect(self.ad_change)

        self.PeakSlide = QtWidgets.QSlider(self.centralwidget)
        self.PeakSlide.setGeometry(QtCore.QRect(130, 931, 301, 21))
        self.PeakSlide.setOrientation(QtCore.Qt.Horizontal)
        self.PeakSlide.setObjectName("PeakSlide")
        self.PeakSlide.valueChanged[int].connect(self.p_change)

        self.MeanSlide = QtWidgets.QSlider(self.centralwidget)
        self.MeanSlide.setGeometry(QtCore.QRect(590, 880, 331, 22))
        self.MeanSlide.setOrientation(QtCore.Qt.Horizontal)
        self.MeanSlide.setObjectName("MeanSlide")
        self.MeanSlide.valueChanged[int].connect(self.mean_change)

        self.horizontalSlider_4 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_4.setGeometry(QtCore.QRect(590, 830, 331, 22))
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")

        self.horizontalSlider_6 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_6.setRange(5, 50)
        self.horizontalSlider_6.setSliderPosition(self.particle_size)
        self.horizontalSlider_6.setGeometry(QtCore.QRect(590, 930, 331, 22))
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_6.setObjectName("horizontalSlider_6")
        self.horizontalSlider_6.valueChanged[int].connect(self.temp_change)

        self.Update = QtWidgets.QPushButton(self.centralwidget)
        self.Update.setGeometry(QtCore.QRect(60, 990, 171, 41))
        self.Update.setObjectName("Update")
        self.Update.clicked.connect(self.update)

        self.ShowContours = QtWidgets.QPushButton(self.centralwidget)
        self.ShowContours.setGeometry(QtCore.QRect(250, 990, 171, 41))
        self.ShowContours.setCheckable(True)
        #self.ShowContours.setChecked(True)
        self.ShowContours.setObjectName("ShowContours")
        #self.ShowContours.checkStateSet(.connect(self.change_bool)

        self.CalculateParticles = QtWidgets.QPushButton(self.centralwidget)
        self.CalculateParticles.setGeometry(QtCore.QRect(120, 1060, 241, 61))
        self.CalculateParticles.setObjectName("CalculateParticles")
        self.CalculateParticles.clicked.connect(self.calcParticles)

        self.ShowOriginal = QtWidgets.QPushButton(self.centralwidget)
        self.ShowOriginal.setGeometry(QtCore.QRect(10, 760, 171, 31))
        self.ShowOriginal.setObjectName("ShowOriginal")

        self.BackgroundLabel = QtWidgets.QLabel(self.centralwidget)
        self.BackgroundLabel.setGeometry(QtCore.QRect(10, 830, 111, 16))
        self.BackgroundLabel.setObjectName("BackgroundLabel")

        self.AdaptiveLabel = QtWidgets.QLabel(self.centralwidget)
        self.AdaptiveLabel.setGeometry(QtCore.QRect(10, 880, 111, 16))
        self.AdaptiveLabel.setObjectName("AdaptiveLabel")

        self.MeanLabel = QtWidgets.QLabel(self.centralwidget)
        self.MeanLabel.setGeometry(QtCore.QRect(460, 880, 101, 16))
        self.MeanLabel.setObjectName("MeanLabel")

        self.PeaksLabel = QtWidgets.QLabel(self.centralwidget)
        self.PeaksLabel.setGeometry(QtCore.QRect(10, 930, 111, 16))
        self.PeaksLabel.setObjectName("PeaksLabel")

        self.ParticleAproxLabel = QtWidgets.QLabel(self.centralwidget)
        self.ParticleAproxLabel.setGeometry(QtCore.QRect(550, 1020, 121, 20))
        self.ParticleAproxLabel.setObjectName("ParticleAproxLabel")

        self.ParticleDistanceLabel = QtWidgets.QLabel(self.centralwidget)
        self.ParticleDistanceLabel.setGeometry(QtCore.QRect(550, 1050, 111, 16))
        self.ParticleDistanceLabel.setObjectName("ParticleDistanceLabel")

        self.ParticleSizeBox = QtWidgets.QTextEdit(self.centralwidget)
        self.ParticleSizeBox.setGeometry(QtCore.QRect(680, 1020, 51, 21))
        self.ParticleSizeBox.setObjectName("ParticleSizeBox")

        self.DistanceBox = QtWidgets.QTextEdit(self.centralwidget)
        self.DistanceBox.setGeometry(QtCore.QRect(680, 1050, 51, 21))
        self.DistanceBox.setObjectName("DistanceBox")

        self.ParticleTemplate = QtWidgets.QLabel(self.centralwidget)
        self.ParticleTemplate.setGeometry(QtCore.QRect(760, 990, 111, 111))
        self.ParticleTemplate.setText("")
        self.ParticleTemplate.setPixmap(QtGui.QPixmap("../../../../Pictures/s-l3"))
        self.ParticleTemplate.setScaledContents(True)
        self.ParticleTemplate.setObjectName("ParticleTemplate")

        self.TemplateLabel = QtWidgets.QLabel(self.centralwidget)
        self.TemplateLabel.setGeometry(QtCore.QRect(770, 1100, 91, 16))
        self.TemplateLabel.setObjectName("TemplateLabel")

        self.LoremIpsum1 = QtWidgets.QLabel(self.centralwidget)
        self.LoremIpsum1.setGeometry(QtCore.QRect(460, 830, 101, 16))
        self.LoremIpsum1.setObjectName("LoremIpsum1")

        self.LoremIpsum2 = QtWidgets.QLabel(self.centralwidget)
        self.LoremIpsum2.setGeometry(QtCore.QRect(460, 930, 101, 16))
        self.LoremIpsum2.setObjectName("LoremIpsum2")

        self.ShowBackground = QtWidgets.QPushButton(self.centralwidget)
        self.ShowBackground.setGeometry(QtCore.QRect(270, 760, 221, 31))
        self.ShowBackground.setObjectName("ShowBackground")

        self.ShowAdaptive = QtWidgets.QPushButton(self.centralwidget)
        self.ShowAdaptive.setGeometry(QtCore.QRect(490, 760, 221, 31))
        self.ShowAdaptive.setObjectName("ShowAdaptive")

        self.ShowPeaks = QtWidgets.QPushButton(self.centralwidget)
        self.ShowPeaks.setGeometry(QtCore.QRect(710, 760, 221, 31))
        self.ShowPeaks.setObjectName("ShowPeaks")

        self.filterLabel = QtWidgets.QLabel(self.centralwidget)
        self.filterLabel.setGeometry(QtCore.QRect(220, 770, 47, 13))
        self.filterLabel.setObjectName("Filter")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 947, 21))
        self.menubar.setObjectName("menubar")

        self.menuinterfaceParticles = QtWidgets.QMenu(self.menubar)
        self.menuinterfaceParticles.setObjectName("menuinterfaceParticles")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuinterfaceParticles.menuAction())
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Update.setText(_translate("MainWindow", "Update Image"))
        self.ShowContours.setText(_translate("MainWindow", "Show Contours"))
        self.CalculateParticles.setText(_translate("MainWindow", "Calculate  Particles"))
        self.ShowOriginal.setText(_translate("MainWindow", "Original"))
        self.BackgroundLabel.setText(_translate("MainWindow", "Background Thresh:"))
        self.AdaptiveLabel.setText(_translate("MainWindow", "Adaptive Thresh:"))
        self.MeanLabel.setText(_translate("MainWindow", "Mean (Reduce):"))
        self.PeaksLabel.setText(_translate("MainWindow", "Gradient Peaks:"))
        self.ParticleAproxLabel.setText(_translate("MainWindow", "Particle Size Aprox. (px)"))
        self.ParticleDistanceLabel.setText(_translate("MainWindow", "Particle Distance (px):"))
        self.ParticleSizeBox.setHtml(_translate("MainWindow",
                                                "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                                "p, li { white-space: pre-wrap; }\n"
                                                "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
                                                "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">20</p></body></html>"))
        self.DistanceBox.setHtml(_translate("MainWindow",
                                            "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                            "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                            "p, li { white-space: pre-wrap; }\n"
                                            "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">14</p></body></html>"))
        self.TemplateLabel.setText(_translate("MainWindow", "Particle Template"))
        self.LoremIpsum1.setText(_translate("MainWindow", "Lorem Ipsum"))
        self.LoremIpsum2.setText(_translate("MainWindow", "Lorem Ipsum"))
        self.ShowBackground.setText(_translate("MainWindow", "Background"))
        self.ShowAdaptive.setText(_translate("MainWindow", "Outer Particle Border"))
        self.ShowPeaks.setText(_translate("MainWindow", "Inner Positions"))
        self.filterLabel.setText(_translate("MainWindow", "Filter:"))
        self.menuinterfaceParticles.setTitle(_translate("MainWindow", "interfaceParticles"))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWin = QtWidgets.QMainWindow()
    ui = ParticleUserinterface(PATH)
    sys.exit(app.exec_())
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys
import cv2

from processing_module import ParticleProcessing
PATH = r'Results/gui_img.jpg'  # Path jpg File

class Ui(QtWidgets.QMainWindow):
    def __init__(self, path):
        super(Ui, self).__init__()
        uic.loadUi('UI/ParticleUI.ui', self)
        self.proc = ParticleProcessing(path)

        self.setupUi()
        self.active = self.proc.img
        self.show_image()
        self.show()

    def setupUi(self):

        # Slider Actions
        self.BackgroundSlide.valueChanged[int].connect(self.bg_change)
        self.AdaptiveSlide.valueChanged[int].connect(self.ad_change)
        self.MeanSlide.valueChanged[int].connect(self.mean_change)
        self.PeakSlide.valueChanged[int].connect(self.p_change)
        self.ParticleSizeSlide.valueChanged[int].connect(self.temp_change)
        #Button Actions
        self.quitButton.clicked.connect(self.quit)
        self.CalculateParticles.clicked.connect(self.proc.calcParticles)
        self.ShowTemplate.clicked.connect(self.show_temp)
        self.ShowBackground.clicked.connect(self.showBgThresh)
        self.ShowAdaptive.clicked.connect(self.showAdThresh)
        self.ShowPeaks.clicked.connect(self.showPeaks)
        self.ShowOriginal.clicked.connect(self.showImage)
        # Textbox
        self.gap_textbox.textChanged[str].connect(self.gapUpdate)

    def show_image(self, img=None):
        if img is not None:
            self.frame = img
        else:
            self.frame = self.active
        self.frame = QtGui.QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0],
                                  QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ImageWin.setPixmap(QtGui.QPixmap.fromImage(self.frame))

    def show_temp(self):
        tmpImg = self.proc.template
        cv2.cvtColor(tmpImg, cv2.COLOR_GRAY2BGR)
        cv2.imshow(' e', tmpImg)

    def showBgThresh(self):
        if self.ShowContours.isChecked():
            res = self.proc.bg_thresh(contours=True)
        else:
            res = self.proc.bg_thresh()
            res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        self.show_image(res)

    def showAdThresh(self):
        if self.ShowContours.isChecked():
            res = self.proc.ad_thresh(contours=True)
        else:
            res = self.proc.ad_thresh()
            res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        self.show_image(res)

    def showPeaks(self):
        if self.ShowContours.isChecked():
            res = self.proc.p_thresh(contours=True)
        else:
            res = self.proc.p_thresh()
            res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        self.show_image(res)

    def showImage(self):
        self.show_image(self.proc.img)

    def bg_change(self, val):
        self.proc.bgThresh = val    #set Value
        self.bg_val.setText(str(val)) # update label
        self.showBgThresh() #show result

    def ad_change(self, val):
        self.proc.blockSize = 2 * val + 1
        self.ad_val.setText(str(self.proc.blockSize))
        self.showAdThresh()

    def p_change(self, val):
        self.proc.peaksThresh = val / 100
        self.p_val.setText(str(self.proc.peaksThresh))
        self.showPeaks()

    def mean_change(self, val):
        self.proc.C = val
        self.mean_val.setText(str(self.proc.C))
        self.showAdThresh()

    def temp_change(self, val):
        self.proc.particle_size = val
        self.proc.DistTempl()
        self.show_temp()

    def gapUpdate(self):
        if self.gap_textbox.text().isnumeric() == True:
            self.proc.gap = int(self.gap_textbox.text())
            self.proc.DistTempl()
        elif self.gap_textbox.text().isnumeric() == False:
            self.gap_textbox.setText('error')
            self.gap_textbox.selectAll()

    def update(self):
        self.proc.process_all()
        self.show_image(self.proc.img)

    def quit(self):
        cv2.destroyAllWindows()
        sys.exit(app.exec_())

app = QtWidgets.QApplication(sys.argv)
window = Ui(PATH)
app.exec_()
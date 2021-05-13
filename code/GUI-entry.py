import sys
import cv2
from PIL import Image
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import numpy as np
from GUI import Ui_MainWindow
from LK import main


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        # link to ui file
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.width = 0
        self.height = 0
        self.xy = []
        self.videoPath = ""
        self.pedestrian_flag = 0

        # connect methods
        self.pushButton.clicked.connect(self.loadFile)
        self.pushButton_2.clicked.connect(self.clearPoints)
        self.pushButton_3.clicked.connect(self.opticalFlow)
        self.radioButton.toggled.connect(self.pedestrian_button)

        self.center()
        self.setWindowTitle("Optical Flow")
        self.setMouseTracking(False)

        # origin white background
        self.frame = cv2.cvtColor(np.ones((360, 640, 3), dtype=np.uint8) * 255, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(self.frame)
        self.pixmap = frame.toqpixmap()
        self.label_3.setPixmap(self.pixmap)

    # read video
    def loadFile(self):
        video, _ = QFileDialog.getOpenFileName(self, 'select video', 'c:\\', 'Image files(*.mp4)')
        self.videoPath = video
        if len(video) > 0:
            cap = cv2.VideoCapture(video)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # ret, frame = cap.read()
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite("./image/first_frame.jpg", frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame = frame
            self.xy = []
            frame = Image.fromarray(frame)
            frame = frame.toqpixmap()
            self.label_3.setPixmap(frame)
            cap.release()

    # decide whether detect pedestrian
    def pedestrian_button(self):
        if self.radioButton.isChecked():
            self.pedestrian_flag = 1
        elif not self.radioButton.isChecked():
            self.pedestrian_flag = 0

    # clean the manual points in the image
    def clearPoints(self):
        self.xy = []
        frame = cv2.imread("./image/first_frame.jpg")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.toqpixmap()
        self.label_3.setPixmap(frame)

    # center the GUI surface
    def center(self):
        window = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        window.moveCenter(center_point)
        self.move(window.topLeft())

    # start optical tracking
    def opticalFlow(self):
        if len(self.videoPath) == 0:
            print("[INFO] please select a video")
        else:
            # user manually select points
            if len(self.xy) > 0:
                x_y = np.array(self.xy)
                x_y[:, 0] = np.round(x_y[:, 0] / 640 * self.width)
                x_y[:, 1] = np.round(x_y[:, 1] / 360 * self.height)
                if self.pedestrian_flag:
                    main(x_y, self.videoPath, 1)
                else:
                    main(x_y, self.videoPath, 0)
            # user does not select points
            else:
                if self.pedestrian_flag:
                    main(np.array(self.xy), self.videoPath, 1)
                else:
                    print("[INFO] If you don't want pedestrian detection, please select at least one point")

    # click left mouse on the image to manually select points
    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            # only can click inside the image
            if 960 >= event.pos().x() >= 320 and 560 >= event.pos().y() >= 200 and np.sum(self.frame) < 176256000:
                # store the points
                x = event.pos().x() - 320
                y = event.pos().y() - 200
                self.xy.append([x, y])
                frame = cv2.imread("./image/first_frame.jpg")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for i in range(len(self.xy)):
                    frame[self.xy[i][1]-1: self.xy[i][1] + 2, self.xy[i][0]-1:self.xy[i][0] + 2, :] = 0
                self.frame = frame
                frame = Image.fromarray(self.frame)
                frame = frame.toqpixmap()
                self.label_3.setPixmap(frame)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

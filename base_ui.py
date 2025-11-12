import sys
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog
import torch
from main_window_ui import Ui_MainWindow
from PySide6.QtGui import QPixmap, QImage
import cv2
from PySide6.QtCore import Qt
from PySide6.QtCore import QTimer


def convert2QImage(img):
    height, width, channel = img.shape
    return QImage(img.data, width, height, width * channel, QImage.Format_RGB888)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.model = torch.hub.load('./', "custom", "runs/train/exp7/weights/best.pt", source="local")
        self.timer = QTimer()
        self.timer.setInterval(1)
        self.bind_slots()
        self.video = None


    def image_pred(self, file_path):
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转成 RGB
        # 推理
        results = self.model(img)
        image = results.render()[0]

        return convert2QImage(image)

    def video_pred(self):
        # img = cv2.imread(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转成 RGB
        # 推理
        ret, frame = self.video.read()
        if not ret:
            self.timer.stop()
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 缩放原始图片到 input QLabel 大小，保持宽高比
            pix_input = QPixmap.fromImage(convert2QImage(frame)).scaled(
                self.input.width(), self.input.height(), Qt.KeepAspectRatio
            )
            self.input.setPixmap(pix_input)
            results = self.model(frame)
            image = results.render()[0]


            # 缩放检测结果图片到 output QLabel 大小，保持宽高比
            pix_output = QPixmap.fromImage(convert2QImage(image)).scaled(
                self.output.width(), self.output.height(), Qt.KeepAspectRatio
            )
            self.output.setPixmap(pix_output)

    def open_image(self):
      print("点击了检测图片")
      self.timer.stop()
      file_path = QFileDialog.getOpenFileName(self, dir="./dataset/images/train", filter="*.jpg;*.png;*.jpeg")
      if file_path[0]:
        file_path = file_path[0]
        qimage = self.image_pred(file_path)

        # 缩放原始图片到 input QLabel 大小，保持宽高比
        pix_input = QPixmap(file_path).scaled(
          self.input.width(), self.input.height(), Qt.KeepAspectRatio
        )
        self.input.setPixmap(pix_input)

        # 缩放检测结果图片到 output QLabel 大小，保持宽高比
        pix_output = QPixmap.fromImage(qimage).scaled(
          self.output.width(), self.output.height(), Qt.KeepAspectRatio
        )
        self.output.setPixmap(pix_output)
    def open_video(self):
        print("点击了检测视频")
        file_path = QFileDialog.getOpenFileName(self, dir="./dataset", filter="*.mp4")
        if file_path[0]:
            file_path = file_path[0]
            self.video = cv2.VideoCapture(file_path)
            self.timer.start()

    def bind_slots(self):
        self.det_img.clicked.connect(self.open_image)
        self.det_video.clicked.connect(self.open_video)
        self.timer.timeout.connect(self.video_pred)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()






















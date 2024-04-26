# Basic gui app to test model on sample images
import sys
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QDesktopWidget
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt


class ImageSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = tf.keras.models.load_model(R'..\Model\the_best_model.h5')

    def initUI(self):
        self.setWindowTitle('Image Selection and Recognition')
        self.resize(1920, 1080)
        self.center()

        # Layout
        layout = QVBoxLayout()

        # Label for displaying the image
        self.imageLabel = QLabel('Image will appear here')
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setScaledContents(False)
        layout.addWidget(self.imageLabel)
        layout.setStretchFactor(self.imageLabel, 5)

        # Button to open the file dialog
        btnOpenFileDialog = QPushButton('Open photo', self)
        btnOpenFileDialog.clicked.connect(self.openFileDialog)
        layout.addWidget(btnOpenFileDialog)

        # Status label at the bottom
        self.statusLabel = QLabel('Select an image to display')
        self.statusLabel.setAlignment(Qt.AlignCenter)
        font = QFont('Arial', 12, QFont.Bold)
        self.statusLabel.setFont(font)
        layout.addWidget(self.statusLabel)

        self.setLayout(layout)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def openFileDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Choose a photo", "",
                                                  "Image Files (*.jpg *.jpeg *.png *.bmp)", options=options)
        if fileName:
            pixmap = QPixmap(fileName)
            self.imageLabel.setPixmap(
                pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.statusLabel.setText(f'Displaying: {fileName}')

            image = self.prepare_image(fileName)
            prediction = self.model.predict(image)
            self.display_prediction(prediction)

    def prepare_image(self, path):
        img = tf.keras.preprocessing.image.load_img(path, target_size=(128,128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def display_prediction(self, prediction):
        class_labels = [
            "A10", "A400M", "AV8B", "B1", "B2", "B52", "C17", "C2", "C5", "E2", "E7",
            "EF2000", "F117", "F14", "F15", "F16", "F18", "F22", "F35", "F4", "J10",
            "J20", "JAS39", "KC135", "Mig31", "Mirage2000", "MQ9", "P3", "Rafale",
            "RQ4", "SR71", "Su24", "Su25", "Su34", "Su57", "Tornado", "Tu160", "Tu95",
            "U2", "US2", "V22", "Vulcan", "XB70", "YF23"
        ]

        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        self.statusLabel.setText(f'Predicted class: {predicted_class}')


app = QApplication(sys.argv)
ex = ImageSelector()
ex.show()
sys.exit(app.exec_())

# Basic gui app to test model on sample images
import sys
import numpy as np
import tensorflow as tf
import os
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QDesktopWidget
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

IMAGE_RECOGNITION_MODEL = R'..\trained_models\image_recognition_model.h5'
IMAGE_DETECTOR_MODEL = R'..\trained_models\image_detector_model.h5'


class ImageSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.recognition_model = tf.keras.models.load_model(IMAGE_RECOGNITION_MODEL)

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

        self.setLayout(layout)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def openFileDialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose a photo", "",
                                                   "Image Files (*.jpg *.jpeg *.png *.bmp)", options=options)
        if file_name:
            # Process the image as needed
            image = self.prepare_image(file_name)
            prediction = self.recognition_model.predict(image)
            self.get_and_display_prediction(prediction)

            # Get predicted class
            predicted_class = self.get_and_display_prediction(prediction)

            # Add text to the image
            temp_output_path = 'temp_image_with_text.png'
            self.add_text_to_image(file_name, predicted_class, temp_output_path)

            # Load the modified image with QPixmap
            pixmap = QPixmap(temp_output_path)
            self.imageLabel.setPixmap(
                pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            # Clean up temporary file
            # if os.path.exists(temp_output_path):
            #     os.remove(temp_output_path)

    def add_text_to_image(self, image_path, text, output_path):
        # Open the original image
        original_image = Image.open(image_path).convert('RGB')
        original_width, original_height = original_image.size

        # Define font size
        font = ImageFont.load_default()

        # Draw the text background onto the new image
        draw = ImageDraw.Draw(original_image)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate the new height for the image
        new_height = original_height + text_height + 20

        # Create a new image with the adjusted height
        new_image = Image.new('RGB', (original_width, new_height), color=(255, 255, 255))
        new_image.paste(original_image, (0, 0))

        # Draw the full-width yellow rectangle
        background_position = (0, original_height, original_width, original_height + text_height + 20)
        draw = ImageDraw.Draw(new_image)
        draw.rectangle(background_position, fill="yellow")

        # Draw the text onto the new image, centered within the yellow background
        text_x_position = (original_width - text_width) // 2
        text_y_position = original_height + 10
        draw.text((text_x_position, text_y_position), text, font=font, fill="black")

        # Save the new image
        new_image.save(output_path)

    def prepare_image(self, path):
        img = tf.keras.preprocessing.image.load_img(path, target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def get_and_display_prediction(self, prediction):
        directory_path = '../data'
        all_entries = os.listdir(directory_path)
        class_labels = [entry for entry in all_entries if os.path.isdir(os.path.join(directory_path, entry))]

        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]

        return predicted_class


app = QApplication(sys.argv)
ex = ImageSelector()
ex.show()
sys.exit(app.exec_())

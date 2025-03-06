import torchvision
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import sys
import numpy as np
import shutil  # Used to move images
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from torchvision.models import regnet_y_16gf, RegNet_Y_16GF_Weights

class ModelLoader:
    def __init__(self, model_path, class_names):
        # Load the model with pre-trained weights
        weights = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1
        self.model = regnet_y_16gf(weights=weights)
        
        # Modify the first conv layer to accept grayscale images (1 channel instead of 3)
        self.model.stem[0] = nn.Conv2d(1, self.model.stem[0].out_channels, kernel_size=self.model.stem[0].kernel_size, 
                                  stride=self.model.stem[0].stride, padding=self.model.stem[0].padding, bias=False)

        # Modify the final fully connected layer to match the number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(class_names))  # Adjust based on number of class names

        # Load custom weights if provided
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))  # Map to CPU or CUDA
        self.model.eval()

        self.class_names = class_names  # Save the provided class names

    def predict(self, image):
        # Preprocess the image for the model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),  # Ensure image is grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize the grayscale image
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
        return self.class_names[predicted.item()]

# PyQt5 GUI MainWindow class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Classifier')
        self.setGeometry(100, 100, 600, 400)

        # Add the class names here
        self.class_names = ['This is a Wild Type Larva', 'This is a Mutant larva']  # Replace with your actual class names
        self.model_loader = ModelLoader(r'D:\Behavioral genetics_V1\Metamorph_scans\WT_vs_nonWT_image_Classification\Dataset\DATAset_V3\ResNext101_Classification_WT_nonWT_V3.pt', self.class_names)

        self.label = QLabel('Select a folder to test images', self)
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 640)

        self.select_button = QPushButton('Select Folder', self)
        self.select_button.clicked.connect(self.select_folder)

        self.prev_button = QPushButton('Previous', self)
        self.prev_button.clicked.connect(self.show_previous_image)

        self.next_button = QPushButton('Next', self)
        self.next_button.clicked.connect(self.show_next_image)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.image_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.image_files = []
        self.current_image_index = -1

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder:
            self.image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('png', 'jpg', 'jpeg', '.tif'))]
            if self.image_files:
                # Automatically process all images once the folder is selected
                self.process_all_images()

    def process_all_images(self):
        """Method to process all images in the selected folder and save them in respective class folders"""
        for image_path in self.image_files:
            image = Image.open(image_path)
            prediction = self.model_loader.predict(image)
            self.save_image(image_path, prediction)
        self.label.setText('All images processed and saved in respective folders.')

    def show_image_at_index(self):
        if 0 <= self.current_image_index < len(self.image_files):
            image_path = self.image_files[self.current_image_index]
            image = Image.open(image_path)
            prediction = self.model_loader.predict(image)
            self.show_image(image, prediction)

    def show_image(self, image, prediction):
        image = image.convert('RGB').resize((640, 640))
        qimage = QImage(np.array(image), image.width, image.height, image.width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)
        self.label.setText(f'Predicted Class: {prediction}')

    def show_next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_image_at_index()

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image_at_index()

    # Method to save the image in the appropriate folder based on the prediction
    def save_image(self, image_path, prediction):
        # Define output directories for Wild Type and Mutant classes
        output_dirs = {
            'This is a Wild Type Larva': r'C:\Users\Pushkar Bansal\Desktop\mock_test_mixed_images\WildType',  # Update with your desired paths
            'This is a Mutant larva': r'C:\Users\Pushkar Bansal\Desktop\mock_test_mixed_images\Mutant'         # Update with your desired paths
        }

        # Ensure the directories exist
        for dir_path in output_dirs.values():
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # Get the directory based on the prediction
        save_dir = output_dirs.get(prediction)

        if save_dir:
            # Copy the image to the corresponding directory
            file_name = os.path.basename(image_path)
            save_path = os.path.join(save_dir, file_name)
            shutil.copy(image_path, save_path)
            print(f'Saved {file_name} to {save_dir}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

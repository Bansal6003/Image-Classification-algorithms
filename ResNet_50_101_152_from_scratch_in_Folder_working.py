import sys
import os
import torch
import torchvision
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from torchvision import transforms
from PIL import Image
from torch import nn
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##############################################################################
#Step 1 Define Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv3 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1, padding = 0),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

##############################################################################
#Step 2 Define ResNet block
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 7):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# ModelLoader class for loading and predicting with the model
class ModelLoader:
    def __init__(self, model_path, class_names):
        # self.model = ResNet(ResidualBlock, [3, 4, 6, 3])#ResNet 50
        # self.model = ResNet(ResidualBlock, [3, 4, 23, 3]) #ResNet 101
        self.model = ResNet(ResidualBlock, [3, 8, 36, 3]) #ResNet 152
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.class_names = class_names

    def predict(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
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
        self.class_names = ['Wild Type', 'Dead Larva_Unknown', 'Edema', 'Edema_TailCurl', 'TailCurl']  # Replace with your actual class names
        self.model_loader = ModelLoader(r"D:\Behavioral genetics_V1\Metamorph_scans\WT_vs_nonWT_image_Classification\Dataset\MultiClass_Dataset_3dpf\resNet152_Cropped_data.pt", self.class_names)

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
                self.current_image_index = 0
                self.show_image_at_index()

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

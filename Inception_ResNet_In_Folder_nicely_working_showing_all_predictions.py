import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QMessageBox, QScrollArea
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PIL import Image
import numpy as np
import cv2
import os
import shutil
from ultralytics import YOLO
from scipy import stats
import pandas as pd
from torchvision import models
import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])


class CustomInceptionV3(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomInceptionV3, self).__init__()
        # Create the InceptionResnetV2 model
        self.inception = timm.create_model('inception_resnet_v2', pretrained=True)
        
        # Modify the final fully connected layer to output the correct number of classes
        self.inception.classif = nn.Linear(self.inception.classif.in_features, num_classes)

    def forward(self, x):
        return self.inception(x)

# Classification Model Loader
class ModelLoader:
    def __init__(self, model_path, class_names, num_classes=7):
        super(ModelLoader, self).__init__()
        self.model_path = model_path
        self.class_names = class_names  # Initialize class names
        self.model = self.load_model(model_path, num_classes)

    def load_model(self, model_path, num_classes=4):
        # Use the CustomInceptionV3 class with specified number of classes
        model = CustomInceptionV3(num_classes=num_classes)  # Set aux_logits=False for inference

        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model

    def predict(self, image):
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            confidence = probabilities[0][predicted.item()].item()

        predicted_class = self.class_names[predicted.item()]
        return predicted_class, confidence



# Image Size Prediction Model
class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Size Measurements')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.label = QLabel(self)
        layout.addWidget(self.label)

        self.uploadButton = QPushButton('Upload Folder and Process', self)
        self.uploadButton.clicked.connect(self.upload_folder)
        layout.addWidget(self.uploadButton)

        self.backButton = QPushButton('Back', self)
        self.backButton.clicked.connect(self.show_previous_image)
        self.backButton.setEnabled(False)
        layout.addWidget(self.backButton)

        self.forwardButton = QPushButton('Forward', self)
        self.forwardButton.clicked.connect(self.show_next_image)
        self.forwardButton.setEnabled(False)
        layout.addWidget(self.forwardButton)

        self.buttonsLayout = QHBoxLayout()
        layout.addLayout(self.buttonsLayout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.images = []
        self.image_paths = []
        self.processed_images = []
        self.contours_data = []
        self.class_buttons = []
        self.models = {}
        self.current_index = -1
        self.current_model = None
        self.thresholds = {}  # Dictionary to hold thresholds for each class

    def load_models(self):
        # Load all models
        model_paths = {
            # 'Eye': r'D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Collective_Trained_models_noBB_with_BB_ZB_images\train_eye_with_ZB_BB_images_noBB\weights\last.pt',
            # 'Head': r'C:\Users\Pushkar Bansal\Desktop\Behavioral genetics\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Trained_aug_models_noBB\train_aug_head_noBB\weights\last.pt',
            # 'Head-Yolk Extension': r'C:\Users\Pushkar Bansal\Desktop\Behavioral genetics\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Trained_aug_models_noBB\train_aug_head_yolkext_noBB\weights\last.pt',
            'Whole Larva':  r"D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Collective_3dpf_models_real_deal\train_whole_larva_3dpf\weights\best.pt",
            # 'Yolk-extension': r'C:\Users\Pushkar Bansal\Desktop\Behavioral genetics\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Trained_aug_models_noBB\train3_48_params_yolext_noBB\weights\last.pt',
            # 'Yolkext-tail': r'D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Collective_Trained_models_noBB_with_BB_ZB_images\train_yolkext_tail_with_ZB_BB_images_noBB\weights\last.pt',
            # 'Yolk-Sac': r'D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Yolov9_V2_WT_MIB_sampled\train_yolk_sac_WT_MIB_samples\weights\last.pt',
            # 'Pericardium': r'D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Collective_Trained_models_noBB_with_BB_ZB_images\train_pericardium_with_ZB_BB_images_noBB\weights\best.pt'
        }

        for class_name, model_path in model_paths.items():
            self.models[class_name] = YOLO(model_path)

    def upload_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder Containing Test Images')
        if not folder_path:
            return
    
        self.load_models()
    
        pixel_size = 0.008  # based on real-world dimensions
    
        self.images = []
        self.image_paths = []
        self.processed_images = []
        self.contours_data = []
        self.current_index = -1
    
        all_contours = {class_name: [] for class_name in self.models.keys()}
    
        # Create an output directory
        output_directory = os.path.join(folder_path, "processed_images")
        os.makedirs(output_directory, exist_ok=True)
    
        # Initialize DataFrame with size and anomaly columns for each class
        self.df = pd.DataFrame(columns=["Image"] + [f'{class_name}_Size' for class_name in self.models.keys()] + [f'{class_name}_Anomaly' for class_name in self.models.keys()])
    
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                image_path = os.path.join(folder_path, filename)
                original_image = cv2.imread(image_path)
                if original_image is not None:
                    self.images.append(original_image)
                    self.image_paths.append(image_path)
    
                    self.process_all_classes(image_path, pixel_size, all_contours, output_directory)
    
        # Compute thresholds based on IQR
        self.thresholds = self.compute_thresholds(all_contours)
    
        # Process again with thresholds and save results
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                image_path = os.path.join(folder_path, filename)
                contour_sizes = self.process_all_classes(image_path, pixel_size, all_contours, output_directory, True)
                self.df = pd.concat([self.df, contour_sizes])
    
        output_file = os.path.join(folder_path, "contour_sizes.xlsx")
        self.df.to_excel(output_file, index=False)
    
        if self.images:
            self.current_index = 0
            self.display_image(self.images[self.current_index])
            self.update_buttons()
            self.create_class_buttons()
    
            QMessageBox.information(self, 'Success', f'Contour sizes saved to {output_file}.')
        else:
            QMessageBox.warning(self, 'Warning', 'No images processed successfully.')


    def process_all_classes(self, image_path, pixel_size, all_contours, output_directory, include_anomaly=False):
        contour_sizes = {"Image": os.path.basename(image_path)}
    
        image_name = os.path.basename(image_path)
        image_without_extension = os.path.splitext(image_name)[0]
    
        has_anomaly = False
        anomaly_classes = []
    
        for class_name, model in self.models.items():
            contours_info = self.process_image(image_path, model, pixel_size)
            total_area = sum([info['area'] for info in contours_info])
            all_contours[class_name].extend([info['area'] for info in contours_info])
    
            # Separate size and anomaly status into different columns
            contour_sizes[class_name + '_Size'] = total_area
    
            if include_anomaly:
                lower_threshold, upper_threshold = self.thresholds.get(class_name, (float('-inf'), float('inf')))
                if total_area < lower_threshold or total_area > upper_threshold:
                    has_anomaly = True
                    anomaly_classes.append(class_name)
                    contour_sizes[class_name + '_Anomaly'] = 'Anomaly'
                else:
                    contour_sizes[class_name + '_Anomaly'] = 'No Anomaly'
            else:
                contour_sizes[class_name + '_Anomaly'] = 'No Anomaly'  # Default to 'No Anomaly'
    
        if include_anomaly:
            if has_anomaly:
                for class_name in anomaly_classes:
                    anomaly_directory = os.path.join(output_directory, class_name)
                    os.makedirs(anomaly_directory, exist_ok=True)
                    cv2.imwrite(os.path.join(anomaly_directory, image_name), cv2.imread(image_path))
            else:
                non_anomalous_directory = os.path.join(output_directory, "No_Anomalies")
                os.makedirs(non_anomalous_directory, exist_ok=True)
                cv2.imwrite(os.path.join(non_anomalous_directory, image_name), cv2.imread(image_path))
    
        return pd.DataFrame([contour_sizes])


    def process_image(self, image_path, model, pixel_size):
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error loading image: {image_path}")
            return []
        
        results = model(image)
        
        contours_info = []
        
        for result in results:
            if result.masks is None:
                continue
            
            segments = result.masks.xy
            names = result.names
            
            for i, segment in enumerate(segments):
                class_id = int(result.boxes[i].cls.cpu().numpy()[0])
                class_name = names[class_id]
                
                contour = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
                
                if contour.size == 0:
                    print(f"Empty contour for class {class_name}")
                    continue
                
                epsilon = 0.0035 * cv2.arcLength(contour, True)
                smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                if smoothed_contour.shape[0] >= 3:
                    area = cv2.contourArea(smoothed_contour) * (pixel_size ** 2)
                    contours_info.append({
                        'class_name': class_name,
                        'contour': smoothed_contour,
                        'area': area
                    })
                else:
                    print(f"Invalid contour shape for class {class_name}")
        
        return contours_info


    def compute_thresholds(self, all_contours):
        thresholds = {}
        for class_name, areas in all_contours.items():
            if len(areas) > 0:
                q1, q3 = np.percentile(areas, [20, 60])
                iqr = q3 - q1
                lower_threshold = q1 - 1.5 * iqr
                upper_threshold = q3 + 1.5 * iqr
                thresholds[class_name] = (lower_threshold, upper_threshold)
        return thresholds
    

    def show_class_contours(self, class_name):
        if not self.images:
            return
        
        self.current_model = self.models[class_name]
        
        image_path = self.image_paths[self.current_index]
        contours_info = self.process_image(image_path, self.current_model, 0.008)
        
        original_image = self.images[self.current_index].copy()
        
        if not contours_info:
            QMessageBox.warning(self, 'No Contours', f'No contours found for class {class_name}')
            return
        
        for contour_info in contours_info:
            contour = contour_info['contour']
            area = contour_info['area']
            
            if contour.shape[0] >= 3:
                cv2.drawContours(original_image, [contour], -1, (0, 255, 0), 2)
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    lower_threshold, upper_threshold = self.thresholds.get(class_name, (float('-inf'), float('inf')))
                    if area < lower_threshold or area > upper_threshold:
                        cv2.putText(original_image, f'{class_name} (anomaly)', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.putText(original_image, class_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        self.display_image(original_image)

    
    def display_image(self, image):
        # Get the dimensions of the window
        window_width = self.label.width()
        window_height = self.label.height()
        
        # Resize the image to fit within the window dimensions, maintaining the aspect ratio
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        aspect_ratio = w / h
    
        if w > window_width or h > window_height:
            if aspect_ratio > 1:
                # Image is wider than tall, resize based on width
                new_width = window_width
                new_height = int(window_width / aspect_ratio)
            else:
                # Image is taller than wide, resize based on height
                new_height = window_height
                new_width = int(window_height * aspect_ratio)
        else:
            new_width, new_height = w, h
    
        resized_image = cv2.resize(image_rgb, (new_width, new_height))
    
        # Convert to QImage and display
        bytes_per_line = ch * new_width
        q_img = QImage(resized_image.data, new_width, new_height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)

    def show_previous_image(self):
        if self.images and self.current_index > 0:
            self.current_index -= 1
            self.display_image(self.images[self.current_index])
            self.update_buttons()

    def show_next_image(self):
        if self.images and self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.display_image(self.images[self.current_index])
            self.update_buttons()

    def update_buttons(self):
        self.backButton.setEnabled(self.current_index > 0)
        self.forwardButton.setEnabled(self.current_index < len(self.images) - 1)

    def create_class_buttons(self):
        for class_name in self.models.keys():
            button = QPushButton(class_name, self)
            button.clicked.connect(lambda checked, name=class_name: self.show_class_contours(name))
            self.buttonsLayout.addWidget(button)
            self.class_buttons.append(button)

    def on_class_button_clicked(self):
        clicked_button = self.sender()
        for button in self.class_buttons:
            button.setChecked(False)
        clicked_button.setChecked(True)
        class_name = clicked_button.text()
        self.current_model = self.models[class_name]

# Main Application
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Danio AI: ')
        self.setGeometry(100, 100, 600, 400)

        self.classification_button = QPushButton('Image Classification', self)
        self.classification_button.clicked.connect(self.open_classification)
        self.classification_button.setGeometry(200, 100, 200, 50)

        self.size_prediction_button = QPushButton('Size Prediction', self)
        self.size_prediction_button.clicked.connect(self.open_size_prediction)
        self.size_prediction_button.setGeometry(200, 200, 200, 50)

    def open_classification(self):
        class_names = ['Edema', 'Edema_TailCurl', 'TailCurl', 'Wild Type']  # Replace with actual class names
        model_path = r"D:\Behavioral genetics_V1\Metamorph_scans\WT_vs_nonWT_image_Classification\Dataset\MultiClass_Dataset_3dpf\Multiimage_dataset_V5\Inception_ResNet_V12_isolated_images_6_classes.pt"  # Replace with actual model path
        self.classification_window = ClassificationWindow(model_path, class_names)
        self.classification_window.show()

    def open_size_prediction(self):
        self.processor = ImageProcessor()
        self.processor.show()
        

# Classification Window
class ClassificationWindow(QMainWindow):
    def __init__(self, model_path, class_names):
        super().__init__()
        self.setWindowTitle('Image Classification')
        self.setGeometry(100, 100, 800, 600)
        self.model_loader = ModelLoader(model_path, class_names)  # ModelLoader should handle loading the model
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Class Predictor")
        self.setGeometry(100, 100, 800, 600)
        
        main_layout = QVBoxLayout()
        
        # Upload button
        self.button = QPushButton("Upload Folder")
        self.button.clicked.connect(self.upload_folder)
        main_layout.addWidget(self.button)
        
        # Scroll area for results
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.results_layout = QVBoxLayout(scroll_content)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        # Create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def upload_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.make_predictions(folder_path)

    def make_predictions(self, folder_path):
        classes = ['Edema', 'Edema_TailCurl', 'TailCurl', 'Wild Type']  # Update with your actual class names
        
        # Clear previous results
        for i in reversed(range(self.results_layout.count())): 
            self.results_layout.itemAt(i).widget().setParent(None)
        
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                
                # Create a horizontal layout for each image-prediction pair
                hbox = QHBoxLayout()
                
                # Load and display the image
                pixmap = QtGui.QPixmap(img_path)
                pixmap = pixmap.scaled(200, 200, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                img_label = QLabel()
                img_label.setPixmap(pixmap)
                hbox.addWidget(img_label)
                
                # Make prediction using the model_loader
                with Image.open(img_path) as img:
                    img_tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = self.model_loader.model(img_tensor)  # Access the model from model_loader
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    pred_prob = probabilities[0][predicted_class].item()
                
                # Display prediction
                pred_label = QLabel(f"File: {filename}\nPredicted: {classes[predicted_class]}\nProbability: {pred_prob:.4f}")
                hbox.addWidget(pred_label)
                
                # Add this image-prediction pair to the results layout
                self.results_layout.addLayout(hbox)
        
        # Add a stretch to push all items to the top
        self.results_layout.addStretch()

    def classify_image(self, image_path):
        image = Image.open(image_path)
        predicted_class, confidence = self.model_loader.predict(image)  # Use model_loader's predict method
        self.display_image(image_path, predicted_class, confidence)
    
    def display_image(self, image_path, predicted_class, confidence):
        image = QPixmap(image_path)
        self.label.setPixmap(image.scaled(self.label.size(), Qt.KeepAspectRatio))
        QMessageBox.information(self, 'Prediction', f'Predicted class: {predicted_class} \nConfidence: {confidence:.2f}')




if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

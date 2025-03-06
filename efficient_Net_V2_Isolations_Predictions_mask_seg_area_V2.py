import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QMessageBox, QScrollArea, QProgressDialog
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
import traceback
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
import os
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])


class CustomEfficientNetV2(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomEfficientNetV2, self).__init__()
        # Create the EfficientNetV2-S model (you can also use -M or -L variants)
        self.efficientnet = timm.create_model('tf_efficientnetv2_s', pretrained=True)
        
        # Modify the classifier head
        num_features = self.efficientnet.classifier.in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

# Classification Model Loader
class ModelLoader:
    def __init__(self, model_path, class_names, num_classes=6):
        super(ModelLoader, self).__init__()
        self.model_path = model_path
        self.class_names = class_names  # Initialize class names
        self.model = self.load_model(model_path, num_classes)

    def load_model(self, model_path, num_classes=6):
      
        model = CustomEfficientNetV2(num_classes=num_classes)  # Set aux_logits=False for inference

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
        class_names = ['Dead Larva_Unknown','Edema', 'Noto', 'Small Eyes', 'TailCurl', 'Wild Type']  # Replace with actual class names
        model_path =  r"D:\Behavioral genetics_V1\Metamorph_scans\WT_vs_nonWT_image_Classification\Dataset\MultiClass_Dataset_3dpf\Multiimage_dataset_V5\Best Models_regularly updated\Classification models\Efficient_NetV2-s_uncropped_6Classes_V5_80_20_split.pt"  # Replace with actual model path
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
        
        # Load the YOLO model
        try:
            self.model = YOLO(r"D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Collective_3dpf_models_real_deal\whole_larva_v2\weights\last.pt")
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to load YOLO model: {e}")
            sys.exit(1)

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
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder Containing Original Test Images')
        if not folder_path:
            print("No folder selected.")
            return

        print(f"Selected folder: {folder_path}")

        self.images = []
        self.image_paths = []
        self.processed_images = []
        self.current_index = -1

        output_directory = os.path.join(folder_path, "processed_images")
        isolated_objects_directory = os.path.join(folder_path, "isolated_objects")
        
        for directory in [output_directory, isolated_objects_directory]:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        print(f"Found {len(image_files)} image files.")

        for filename in image_files:
            image_path = os.path.join(folder_path, filename)
            print(f"Processing image: {image_path}")
            
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"Failed to read image: {image_path}")
                continue

            self.image_paths.append(image_path)

            try:
                # Process the image
                results = self.model(image_path, show=False, save=False)
                predicted_img = results[0].plot()
                print(f"YOLO processing completed for {filename}")

                # Extract and save isolated objects
                self.extract_objects(original_image, results[0], isolated_objects_directory, filename)

                # Resize the image for display
                resized_img = cv2.resize(predicted_img, 800, 600)
                self.processed_images.append(resized_img)

                # Save the processed image with bounding boxes
                output_image_path = os.path.join(output_directory, f'processed_{filename}')
                cv2.imwrite(output_image_path, predicted_img)
                print(f"Saved processed image with bounding boxes: {output_image_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                print(traceback.format_exc())

        if self.processed_images:
            self.current_index = 0
            self.display_image(self.processed_images[self.current_index])
            self.update_navigation_buttons()
            QMessageBox.information(self, 'Success', 'Images processed and objects isolated.')
        else:
            QMessageBox.warning(self, 'Warning', 'No images were processed.')
        
        folder_path = QFileDialog.getExistingDirectory(self, "Select folder containing isolated images")
        if folder_path:
            self.make_predictions(folder_path)

    def extract_objects(self, image, result, output_directory, filename):
        print(f"Extracting objects from {filename}")
        print(f"Result type: {type(result)}")
        print(f"Result attributes: {dir(result)}")
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            print(f"Number of detected objects: {len(boxes)}")
            
            for i, box in enumerate(boxes):
                try:
                    print(f"Processing box {i+1}")
                    print(f"Box type: {type(box)}")
                    print(f"Box attributes: {dir(box)}")
                    
                    if hasattr(box, 'xyxy'):
                        # Get the coordinates of the bounding box
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        print(f"Object {i+1} coordinates: ({x1}, {y1}, {x2}, {y2})")
                        
                        # Extract the object from the image
                        object_img = image[y1:y2, x1:x2]
                        
                        if object_img.size > 0:  # Check if the extracted image is not empty
                            # Save the extracted object
                            obj_filename = f'object_{os.path.splitext(filename)[0]}_{i}.png'
                            obj_path = os.path.join(output_directory, obj_filename)
                            success = cv2.imwrite(obj_path, object_img)
                            if success:
                                print(f"Saved isolated object: {obj_path}")
                            else:
                                print(f"Failed to save isolated object: {obj_path}")
                                print(f"Object image shape: {object_img.shape}")
                                print(f"Object image dtype: {object_img.dtype}")
                        else:
                            print(f"Warning: Empty extracted image for object {i+1}")
                    else:
                        print(f"Box {i+1} does not have 'xyxy' attribute")
                except Exception as e:
                    print(f"Error extracting object {i+1}: {e}")
                    print(traceback.format_exc())
        else:
            print("No boxes found in the result")
    
    # def resize_image(self, image, new_width, new_height):
    #     return cv2.resize(image, (new_width, new_height))
    
    def make_predictions(self, folder_path):
        classes = ['Dead Larva_Unknown','Edema', 'Noto', 'Small Eyes', 'TailCurl', 'Wild Type']  # Update with your actual class names
        
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


# Image Size Prediction Model
class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pixel_size = 0.01  # mm per pixel
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Size Measurements - DeepLabV3+')
        self.setGeometry(100, 100, 1000, 800)  # Made window larger
    
        layout = QVBoxLayout()
    
        # Image display label
        self.imageLabel = QLabel(self)
        layout.addWidget(self.imageLabel)
    
        # Upload button
        self.uploadButton = QPushButton('Upload Folder and Process', self)
        self.uploadButton.clicked.connect(self.upload_folder)
        layout.addWidget(self.uploadButton)
    
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.backButton = QPushButton('Previous Image', self)
        self.backButton.clicked.connect(self.show_previous_image)
        self.backButton.setEnabled(False)
        nav_layout.addWidget(self.backButton)
    
        self.forwardButton = QPushButton('Next Image', self)
        self.forwardButton.clicked.connect(self.show_next_image)
        self.forwardButton.setEnabled(False)
        nav_layout.addWidget(self.forwardButton)
        
        layout.addLayout(nav_layout)
    
        # Class buttons layout
        self.class_buttons_layout = QHBoxLayout()
        layout.addLayout(self.class_buttons_layout)
        self.class_buttons = []
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
        self.models = {}
        self.current_index = -1
        self.current_class = None
        self.images = []
        self.image_paths = []
        self.predictions = {}  # Store predictions for each class
        self.results = []

    def load_multiple_models(self):
        model_paths = {
            'Eye': r"D:\Behavioral genetics_V1\Metamorph_scans\Workflow_codes_V2\Pretrained_and_non_custom_algorithms\Deep_Lab\Trained_Models\eye_deeplabv3plus_final_model.pth",
            'Whole Larva': r"D:\Behavioral genetics_V1\Metamorph_scans\Workflow_codes_V2\Pretrained_and_non_custom_algorithms\Deep_Lab\Trained_Models\Whole_Larva_deeplabv3plus_final_model.pth"
        }

        for class_name, model_path in model_paths.items():
            self.models[class_name] = self.load_model(model_path)
            print(f"Loaded {class_name} model")

    def load_model(self, model_path):
        model = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
        
        checkpoint = torch.load(model_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        return model

    def create_class_buttons(self):
        # Clear existing buttons
        for button in self.class_buttons:
            button.setParent(None)
        self.class_buttons.clear()
    
        # Create new buttons for each class
        for class_name in self.models.keys():
            button = QPushButton(class_name, self)
            button.setCheckable(True)  # Make button toggleable
            button.clicked.connect(lambda checked, name=class_name: self.on_class_button_clicked(name))
            self.class_buttons_layout.addWidget(button)
            self.class_buttons.append(button)
    
        # Select first class by default if available
        if self.class_buttons:
            self.class_buttons[0].setChecked(True)
            self.current_class = list(self.models.keys())[0]
    
    def on_class_button_clicked(self, class_name):
        # Update button states
        for button in self.class_buttons:
            button.setChecked(button.text() == class_name)
        
        self.current_class = class_name
        # Update display with current class predictions
        if self.current_index >= 0:
            self.display_current_image()
    
    def display_current_image(self):
        if self.current_index < 0 or not self.current_class:
            return
    
        # Load current image from path
        try:
            image_path = self.image_paths[self.current_index]
            current_image = cv2.imread(image_path)
            if current_image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            current_prediction = self.predictions[self.current_index].get(self.current_class)
            
            if current_prediction is None:
                return
    
            # Convert BGR to RGB
            current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
    
            # Create overlay
            overlay = np.zeros_like(current_image)
            overlay[current_prediction > 0] = [255, 0, 0]  # Red for segmented regions
            blended = cv2.addWeighted(current_image, 0.7, overlay, 0.3, 0)
    
            # Calculate area
            area_mm2 = self.calculate_area(current_prediction, self.pixel_size)
    
            # Add text to image
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"{self.current_class} Area: {area_mm2:.2f} mm²"
            cv2.putText(blended, text, (10, 30), font, 1, (255, 255, 255), 2)
    
            # Convert to QImage and display
            height, width = blended.shape[:2]
            bytes_per_line = 3 * width
            q_img = QImage(blended.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Scale image to fit label
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(
                self.imageLabel.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.imageLabel.setPixmap(scaled_pixmap)
    
        except Exception as e:
            print(f"Error displaying image: {str(e)}")
    
    def upload_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if not folder_path:
            return
    
        try:
            self.load_multiple_models()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load models: {str(e)}")
            return
    
        output_dir = os.path.join(folder_path, "segmentation_results")
        os.makedirs(output_dir, exist_ok=True)
    
        # Clear previous data
        self.images = []
        self.image_paths = []
        self.predictions = {}
        self.results = []
        self.current_index = -1
    
        results = []
    
        # Get list of valid image files first
        valid_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
        # Show progress dialog
        progress = QProgressDialog("Processing images...", "Cancel", 0, len(valid_files), self)
        progress.setWindowModality(Qt.WindowModal)
    
        for idx, filename in enumerate(valid_files):
            if progress.wasCanceled():
                break
    
            progress.setValue(idx)
            image_path = os.path.join(folder_path, filename)
            image_name = os.path.splitext(filename)[0]
    
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
    
                # Store only the image path, not the actual image
                self.image_paths.append(image_path)
                self.predictions[idx] = {}
    
                # Process with each model
                image_results = {'Image': filename}
                input_tensor, original_image, original_size = self.preprocess_image(image_path)
    
                for class_name, model in self.models.items():
                    try:
                        # Get prediction
                        with torch.cuda.amp.autocast():
                            prediction = self.predict(model, input_tensor, original_size)
                        
                        self.predictions[idx][class_name] = prediction
                        
                        # Calculate area
                        area = self.calculate_area(prediction, self.pixel_size)
                        image_results[f'{class_name}_Area'] = area
    
                        # Save visualization
                        vis_path = os.path.join(output_dir, f"{image_name}_{class_name}_viz.png")
                        self.visualize_results(original_image, prediction, vis_path, self.pixel_size)
                        
                        # Clear CUDA cache
                        torch.cuda.empty_cache()
    
                    except Exception as e:
                        print(f"Error processing {filename} with {class_name} model: {str(e)}")
                        continue
    
                results.append(image_results)
    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
            
            # Load first image for display
            if idx == 0:
                self.images.append(image)
    
        progress.setValue(len(valid_files))
    
        # Save results to Excel
        if results:
            df = pd.DataFrame(results)
            output_file = os.path.join(folder_path, "segmentation_measurements.xlsx")
            df.to_excel(output_file, index=False)
    
        if self.image_paths:
            self.current_index = 0
            self.create_class_buttons()
            self.display_current_image()
            self.update_navigation_buttons()
            QMessageBox.information(self, 'Success', 
                                  f'Processing complete. Results saved to {output_file}')

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('L')
        original_size = image.size
        
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        input_tensor = transform(image)
        input_tensor = input_tensor.repeat(3, 1, 1)
        input_tensor = input_tensor.unsqueeze(0)
        
        return input_tensor, image, original_size

    def predict(self, model, input_tensor, original_size):
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            output = model(input_tensor)
            probabilities = torch.sigmoid(output)
            prediction = (probabilities > 0.5).float().squeeze().cpu().numpy()
            
            prediction_img = Image.fromarray((prediction * 255).astype(np.uint8))
            prediction_img = prediction_img.resize(original_size, Image.NEAREST)
            prediction = np.array(prediction_img) > 127
            
        return prediction.astype(np.uint8)

    def calculate_area(self, prediction, pixel_size):
        num_pixels = np.sum(prediction > 0)
        area_mm2 = num_pixels * (pixel_size ** 2)
        return area_mm2

    def visualize_results(self, original_image, prediction, save_path=None, pixel_size=0.01):
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

        area_mm2 = self.calculate_area(prediction, pixel_size)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.title("Original Image")
        plt.imshow(original_image, cmap='gray')
        plt.axis('off')
        
        plt.subplot(132)
        plt.title(f"Mask\nArea: {area_mm2:.2f} mm²")
        plt.imshow(prediction, cmap='gray')
        plt.axis('off')
        
        plt.subplot(133)
        plt.title("Overlay")
        
        overlay = np.zeros_like(original_image)
        overlay[prediction > 0] = [255, 0, 0]
        blended = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
        
        plt.imshow(blended)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
        else:
            plt.show()

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()  # Change from display_image to display_current_image
            self.update_navigation_buttons()
    
    def show_next_image(self):
        if self.current_index < len(self.image_paths) - 1:  # Change from images to image_paths
            self.current_index += 1
            self.display_current_image()  # Change from display_image to display_current_image
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
       self.backButton.setEnabled(self.current_index > 0)
       self.forwardButton.setEnabled(self.current_index < len(self.image_paths) - 1)

    
        


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

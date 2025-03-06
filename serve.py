from fastapi import FastAPI, File, UploadFile, HTTPException
import mlflow.pytorch
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI()

# Initialize model variable
model = None

# Define your class names in the correct order
CLASS_NAMES = ['Dead Larva_Unknown','Edema', 'Noto', 'Small Eyes', 'TailCurl', 'Wild Type']  # Replace with your actual class names

# Set MLflow tracking URI and load model at startup
@app.on_event("startup")
async def startup_event():
    global model
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    try:
        model = mlflow.pytorch.load_model("models:/EfficientNetV2/6")
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Transform image
        image_tensor = transform(image).unsqueeze(0)
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            model.cuda()
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[0][predicted_idx].item()
        
        # Get class name instead of index
        predicted_class = CLASS_NAMES[predicted_idx]
        
        # Return all probabilities with class names
        class_probabilities = {
            class_name: float(prob)
            for class_name, prob in zip(CLASS_NAMES, probabilities[0].tolist())
        }
        
        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "all_probabilities": class_probabilities
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="127.0.0.1", port=8000, reload=True)
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="House Room Classifier", layout="wide", page_icon="🏠")

# --- TITLE & DESCRIPTION ---
st.title("🏠 House Room Image Classifier")
st.write("Upload an image of a room, select a model, and let AI predict the room type!")

# --- MODEL & PARAMETERS ---
CLASSES = ["Livingroom", "Kitchen", "Dinning", "Bedroom", "Bathroom"]
MODEL_PATHS = {
    "SimpleCNN": ".\models\cnn_model .pth",
    "ResNet18": "./models/resnet_model.pth"
}

# --- LOAD MODELS ---
@st.cache_resource
def load_model(model_name):
    if model_name == "SimpleCNN":
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes):
                super(SimpleCNN, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                self.classifier = nn.Sequential(
                    nn.Linear(64 * 56 * 56, 256),
                    nn.ReLU(),
                    nn.Linear(256, num_classes)
                )
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        model = SimpleCNN(len(CLASSES))
    else:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    
    model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location=torch.device('cpu')))
    model.eval()
    return model

# --- IMAGE TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox("Choose Model", ["SimpleCNN", "ResNet18"])
uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# --- PREDICTION FUNCTION ---
def predict(model, image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return CLASSES[predicted.item()]

# --- DISPLAY IMAGE AND PREDICTION ---
if uploaded_image:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("🔍 Prediction Result")
        if st.button("📸 Predict"):
            with st.spinner("🔄 Predicting... Please wait."):
                model = load_model(model_choice)
                image = Image.open(uploaded_image).convert('RGB')
                prediction = predict(model, image)
                st.success(f"🏆 Predicted Room Type: **{prediction}**")
                
                st.write("**Model Used:**", model_choice)
                
                st.markdown(
                    f"""
                    - 📊 **Available Models:** SimpleCNN, ResNet18  
                    - 🏠 **Classes:** {', '.join(CLASSES)}  
                    """
                )

# --- FOOTER ---
st.markdown("---")
st.write("📌 Developed by Adelta Aurelianti | 🧠 AI-Powered Room Classifier | 🚀 Powered by PyTorch & Streamlit")
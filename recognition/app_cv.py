import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import urllib.request

st.set_page_config(page_title="Flower Species Classifier", page_icon="🌸")
st.title("🌸 Flower Species Classifier")

# Load official Flowers-102 class mapping (JSON maps "1".."102" to species names)
@st.cache_data
def load_class_names():
    url = "https://raw.githubusercontent.com/Anish9901/Flowers-102-PyTorch/master/cat_to_name.json"
    req = urllib.request.urlopen(url)
    return json.loads(req.read().decode('utf-8'))

cat_to_name = load_class_names()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 102)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('flower_resnet50.pth', map_location=device))
    model.eval()
    return model, device

try:
    model, device = load_model()
except Exception as e:
    st.error("Could not load 'flower_resnet50.pth'. Ensure train_cv_model.py finished running.")
    st.stop()

uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with st.spinner("Analyzing image features..."):
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top3_prob, top3_catid = torch.topk(probabilities, 3)

    st.subheader("Prediction Results")
    
    # Map index back to 1-based JSON key
    top1_idx = str(top3_catid[0].item() + 1)
    top1_name = cat_to_name.get(top1_idx, "Unknown Flower").title()
    top1_score = top3_prob[0].item() * 100
    
    st.success(f"**Top Prediction:** {top1_name} ({top1_score:.1f}% confidence)")
    
    st.write("---")
    st.write("**Top 3 Likely Species:**")
    for i in range(3):
        idx = str(top3_catid[i].item() + 1)
        name = cat_to_name.get(idx, "Unknown Flower").title()
        prob = top3_prob[i].item() * 100
        st.write(f"- **{name}**: {prob:.1f}%")
        st.progress(int(prob))
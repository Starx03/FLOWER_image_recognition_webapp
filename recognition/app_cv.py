from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

st.set_page_config(page_title="Flower Species Classifier", page_icon="🌸")
st.title("🌸 Flower Species Classifier")

# Official Oxford Flowers-102 Class Mapping
FLOWERS_102_CLASSES = {
    "1": "pink primrose",
    "2": "hard-leaved pocket orchid",
    "3": "canterbury bells",
    "4": "sweet pea",
    "5": "english marigold",
    "6": "tiger lily",
    "7": "moon orchid",
    "8": "bird of paradise",
    "9": "monkshood",
    "10": "globe thistle",
    "11": "snapdragon",
    "12": "colt's foot",
    "13": "king protea",
    "14": "spear thistle",
    "15": "yellow iris",
    "16": "globe-flower",
    "17": "purple coneflower",
    "18": "peruvian lily",
    "19": "balloon flower",
    "20": "giant white arum lily",
    "21": "fire lily",
    "22": "pincushion flower",
    "23": "fritillary",
    "24": "red ginger",
    "25": "grape hyacinth",
    "26": "corn poppy",
    "27": "prince of wales feathers",
    "28": "stemless gentian",
    "29": "artichoke",
    "30": "sweet william",
    "31": "carnation",
    "32": "garden phlox",
    "33": "love in a mist",
    "34": "mexican petunia",
    "35": "alpine sea holly",
    "36": "ruby-lipped cattleya",
    "37": "cape flower",
    "38": "great masterwort",
    "39": "siam tulip",
    "40": "lenten rose",
    "41": "barbeton daisy",
    "42": "daffodil",
    "43": "sword lily",
    "44": "poinsettia",
    "45": "bolero deep blue",
    "46": "wallflower",
    "47": "marigold",
    "48": "buttercup",
    "49": "oxeye daisy",
    "50": "common dandelion",
    "51": "petunia",
    "52": "wild pansy",
    "53": "primula",
    "54": "sunflower",
    "55": "pelargonium",
    "56": "bishop of llandaff",
    "57": "gaura",
    "58": "geranium",
    "59": "orange dahlia",
    "60": "pink-yellow dahlia",
    "61": "cautleya spicata",
    "62": "japanese anemone",
    "63": "black-eyed susan",
    "64": "silverbush",
    "65": "californian poppy",
    "66": "osteospermum",
    "67": "spring crocus",
    "68": "bearded iris",
    "69": "windflower",
    "70": "tree poppy",
    "71": "gazania",
    "72": "azalea",
    "73": "water lily",
    "74": "rose",
    "75": "thorn apple",
    "76": "morning glory",
    "77": "passion flower",
    "78": "lotus",
    "79": "toad lily",
    "80": "anthurium",
    "81": "frangipani",
    "82": "clematis",
    "83": "hibiscus",
    "84": "columbine",
    "85": "desert-rose",
    "86": "tree mallow",
    "87": "magnolia",
    "88": "cyclamen",
    "89": "watercress",
    "90": "canna lily",
    "91": "hippeastrum",
    "92": "bee balm",
    "93": "ball moss",
    "94": "foxglove",
    "95": "bougainvillea",
    "96": "camellia",
    "97": "mallow",
    "98": "mexican petunia",
    "99": "bromelia",
    "100": "blanket flower",
    "101": "trumpet creeper",
    "102": "blackberry lily",
}

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model():
  model = models.resnet50(weights=None)
  model.fc = nn.Linear(model.fc.in_features, 102)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.load_state_dict(
      torch.load("flower_resnet50.pth", map_location=device)
  )
  model.eval()
  return model, device


try:
  model, device = load_model()
except Exception as e:
  st.error(
      "Could not load 'flower_resnet50.pth'. Ensure train_cv_model.py finished"
      " running."
  )
  st.stop()

uploaded_file = st.file_uploader(
    "Choose a flower image...", type=["jpg", "jpeg", "png"]
)

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

  top1_cat_id = str(top3_catid[0].item() + 1)
  top1_name = FLOWERS_102_CLASSES.get(top1_cat_id, "Unknown Flower").title()
  top1_score = top3_prob[0].item() * 100

  st.success(f"**Top Prediction:** {top1_name} ({top1_score:.1f}% confidence)")

  st.write("---")
  st.write("**Top 3 Likely Species:**")
  for i in range(3):
    cat_id = str(top3_catid[i].item() + 1)
    name = FLOWERS_102_CLASSES.get(cat_id, "Unknown Flower").title()
    prob = top3_prob[i].item() * 100
    st.write(f"- **{name}**: {prob:.1f}%")
    st.progress(int(prob))

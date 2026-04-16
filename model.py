import torch
from torchvision import transforms
from PIL import Image

# Classes
class_names = ['benign', 'malignant']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Model load
device = torch.device("cpu")
model = torch.load("/home/skinapp/SKIN_CANCER_APP/model.pt", map_location=device, weights_only=False)
model.eval()

def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item()
    return class_names[predicted], round(confidence, 2)
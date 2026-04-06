import torch
import torch.nn as nn
from torchvision import models
import os

path = os.path.join(os.getcwd(), 'outputs', 'demo_models', 'efficientnet_v2_s_fold_1_best.pt')
print("Checking path:", path)
print("Exists:", os.path.exists(path))

model = models.efficientnet_v2_s(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 2))

try:
    state_dict = torch.load(path, map_location='cpu', weights_only=False)
    print("Loaded dict type:", type(state_dict))
    if isinstance(state_dict, dict):
        print("KEYS in loaded dict (first 5):", list(state_dict.keys())[:5])
    else:
        # It might be a full model object if saved with torch.save(model, ...)
        print("It's a full model object")
        model = state_dict
    
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)
        print('LOADED STATE DICT SUCCESSFULLY')
except Exception as e:
    import traceback
    traceback.print_exc()
    print('FAILED:', e)

from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

model.eval()

for cls in ['normal', 'pancreatic_tumor']:
    folder = os.path.join(os.getcwd(), 'DATASET', 'test', 'test', cls)
    if os.path.exists(folder):
        files = os.listdir(folder)[:5]
        for f in files:
            img_path = os.path.join(folder, f)
            img = Image.open(img_path).convert('RGB')
            tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)
                print(f"[{cls}] {f}: prob tumor = {probs[0][1].item():.4f}")

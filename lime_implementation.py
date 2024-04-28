# Code for this implementation of LIME (marcotcr, 2021) has been developed by following examples and instructions
# defined in its original documentation (sytelus, 2019)
# This implementation of LIME has been modified to work with my own model from Task 2 of my 
# INM702 Coursework Assignment
#
# The dataset used in this implementation is the Animal Species Classification - V3 dataset
# available on Kaggle (DeepNets, Verma, Jain, Agrawal, 2023)

import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
import pickle
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f'Device: {device}')

print(f'loading model')
with open(f'./saved_model.pkl', 'rb') as file:
    model = pickle.load(file)

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 

classes = ['Beetle', 'Butterfly', 'Cat', 'Cow', 'Dog', 'Elephant', 'Gorilla', 'Hippo', 'Lizard', 'Monkey', 'Mouse', 'Panda', 'Spider', 'Tiger', 'Zebra']

#img = get_image('./archive/dataset/Testing Data/Panda/Panda-Test (58).jpeg')
#img = get_image('./archive/dataset/Testing Data/Gorilla/Gorilla (15).jpeg')
#img = get_image('./archive/dataset/Testing Data/Lizard/Lizard-Testing (60).jpg')
img = get_image('./archive/dataset/Testing Data/Cat/cat-test (300).jpeg')
#img = get_image('./archive/dataset/Testing Data/Beetle/Beatle-Test (65).jpeg')

plt.imshow(img)
plt.title('Original Image')
plt.savefig('image.png')
plt.close()

def transform_convert_to_tensor():
    image = transforms.Compose([transforms.ToTensor()])    

    return image

transform_to_tensor = transform_convert_to_tensor()

def lime_model(images):
    model.eval()
    features = torch.stack(tuple(transform_to_tensor(image) for image in images), dim = 0)

    model.to(device)
    features = features.to(device)
    
    logits = model(features)
    predictions = F.softmax(logits, dim = 1)

    return predictions.detach().cpu().numpy()

def resize_transform_mage(): 
    image = transforms.Compose([transforms.Resize((87, 87))])

    return image

resize_img = resize_transform_mage()

test_pred = lime_model([resize_img(img)])
test_pred.squeeze().argmax()

print(f'Predictions:\n{np.round(test_pred[0] * 100, 2)}')
print(f'Predicted Animal: {classes[test_pred.argmax()]} (Confidence: {np.round(test_pred.max() * 100, 2)}%)')

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(resize_img(img)), lime_model, hide_color = 0)

# Show which areas of the image contributed to the prediction of the model
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
encouraging_prediction = mark_boundaries(temp/255.0, mask)
plt.imshow(encouraging_prediction)
plt.title(f'Predicted Animal: {classes[test_pred.argmax()]}, Confidence: {np.round(test_pred.max() * 100, 2)}%')
plt.savefig('Encouraging Preds.png')
plt.close()

# Hightlight which areas of the image contributed to postively to the prediction, and which contributed negatively
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
pros_cons = mark_boundaries(temp/255.0, mask)
plt.imshow(pros_cons)
plt.title(f'Predicted Animal: {classes[test_pred.argmax()]}, Confidence: {np.round(test_pred.max() * 100, 2)}%')
plt.savefig('Pros and Cons Bounds.png')
plt.close()

# Only show areas of the image the model thought was relevant to the prediction
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)
only_relevant_bounds = mark_boundaries(temp/255.0, mask)
plt.imshow(only_relevant_bounds)
plt.title(f'Predicted Animal: {classes[test_pred.argmax()]}, Confidence: {np.round(test_pred.max() * 100, 2)}%')
plt.savefig('Only Relevant Bounds for Preds.png')
plt.close()
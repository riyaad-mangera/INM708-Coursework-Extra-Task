import os    
os.environ["KMP_DUPLICATE_LIB_OK"] = "True" #Kernal would repeatedly die when using matplotlib unless this line was used

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder

from datetime import datetime
import time

import pickle

from model import ImageNeuralNetwork

from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np

from PIL import Image

if torch.cuda.is_available():
    
    device = torch.device("cuda")
    
else:
    
    device = torch.device("cpu")
    
print(f'Device: {device}')

def accuracy(outputs, labels):
        
        a, predictions = torch.max(outputs, dim = 1)
        
        return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))

def evaluate(model, valid_batches):
    
    with torch.no_grad():
        model.eval()
        outputs = []

        print(f'\tEvaluating {len(valid_batches)} batches')
        
        for batch in valid_batches:

            images, labels = batch 
            images, labels = images.to(device), labels.to(device)
            
            predictions = model(images)
            loss = F.cross_entropy(predictions, labels)
            acc = accuracy(predictions, labels)
            
            outputs.append({"valid_loss": loss.detach(), "valid_accuracy": acc})
    
        #Combine losses and accuracies
        batch_loss = [x["valid_loss"] for x in outputs]
        batch_accuracy = [x["valid_accuracy"] for x in outputs]
        
        epoch_loss = torch.stack(batch_loss).mean()
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        
        return {"valid_loss": epoch_loss.item(), "valid_accuracy": epoch_accuracy.item()}

def train(model, train_batches, valid_batches):

    model.to(device)

    history = []

    for epoch in range(model.epochs):
        
        start = datetime.now()
        
        model.train()
        train_losses = []
        train_accuracies = []

        for idx, batch in enumerate(train_batches):

            print(f'Epoch {epoch} Batch {idx} of {len(train_batches)}')

            images, labels = batch 
            images, labels = images.to(device), labels.to(device)
            
            model.zero_grad()

            predictions = model(images)

            predictions = predictions.to(device)

            loss = F.cross_entropy(predictions, labels.long())
            acc = accuracy(predictions, labels)
            
            train_losses.append(loss)
            train_accuracies.append(acc)
            
            loss.backward()
            
            model.optimiser.step()
            model.optimiser.zero_grad()

        result = evaluate(model, valid_batches)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["train_accuracy"] = torch.stack(train_accuracies).mean().item()
        
        print("Epoch [{}], train_loss: {:.4f}, valid_loss: {:.4f}, train_accuracy: {:.4f}, valid_accuracy: {:.4f}".format(
            epoch, result["train_loss"], result["valid_loss"], result["train_accuracy"], result["valid_accuracy"]))
        print(f"Time taken for epoch {epoch}: {datetime.now()-start}")
        
        history.append(result)

    return history

# Training for this model was done using the reduced_dataset_size folder
# Please use this folder if you would like to achieve similar results to my tests
train_path = r"archive\reduced_dataset_size\images_E_S_SB_69x69_a_03\images_E_S_SB_69x69_a_03_train"
test_path = r"archive\reduced_dataset_size\images_E_S_SB_69x69_a_03\images_E_S_SB_69x69_a_03_test"

dataset = ImageFolder(train_path, transforms.Compose([transforms.Resize((69,69)), transforms.ToTensor()]))

test_dataset = ImageFolder(test_path, transforms.Compose([transforms.Resize((69,69)), transforms.ToTensor()]))

img, label = dataset[0]

print(img.shape,label)
print("Classes: \n",dataset.classes)

def display_image(img, label):
    
    print(f"Label : {dataset.classes[label]}")
    
    plt.imshow(img.permute(1,2,0))

# display_image(*dataset[0])

valid_size = int(len(dataset) * 0.2)
train_size = len(dataset) - valid_size 

train_data, valid_data = random_split(dataset, [train_size, valid_size])

print("Train Data:", len(train_data))
print("Validation Data:",  len(valid_data))

batch_size = 64

train_batches = DataLoader(train_data, batch_size, shuffle = True, num_workers = 0, pin_memory = True, drop_last = True)
valid_batches = DataLoader(valid_data, batch_size * 2, num_workers = 0, pin_memory = True, drop_last = True)
test_batches = DataLoader(test_dataset, batch_size, shuffle = True, num_workers = 0, pin_memory = True, drop_last = True)

print(type(train_data))

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_batch(batch):
    
    for images, labels in batch:
        fig,ax = plt.subplots(figsize = (16, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow = 8).permute(1, 2, 0))
        break
        
# show_batch(train_batches)

learning_rate = 0.001
epochs = 25

model = ImageNeuralNetwork(device=device, learning_rate=learning_rate, epochs=epochs)

#optimiser = torch.optim.SGD(model.parameters(), learning_rate, momentum = 0.9, weight_decay = 0.001)
optimiser = torch.optim.Adam(model.parameters(), learning_rate, weight_decay = 0.0001)

model.set_optimiser(optimiser)

print(model)

model.to(device)

# history = train(model, train_batches, valid_batches)

# print("saving model")
# with open(f'./trained_model_2.pkl', 'wb') as file:
#         pickle.dump(model, file)

print(f'loading model')
with open(f'./trained_model_2.pkl', 'rb') as file:
    model = pickle.load(file)

model.to(device)

# prediction = evaluate(model, test_batches)

# print("test_loss: {:.4f}, test_accuracy: {:.4f}".format(prediction["valid_loss"], prediction["valid_accuracy"]))

# train_losses = []
# train_accuracies = []
# valid_losses = []
# valid_accuracies = []


# for idx in range(len(history)):
    
#     train_losses.append(history[idx]["train_loss"])
#     train_accuracies.append(history[idx]["train_accuracy"])
#     valid_losses.append(history[idx]["valid_loss"])
#     valid_accuracies.append(history[idx]["valid_accuracy"])

# plt.plot(train_losses, label = "Training loss")
# plt.plot(valid_losses, label = "Validation loss")
# plt.title("Average loss per epoch with VGG and SGD")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend(frameon = False)
# plt.savefig("loss_2.png")
# #plt.show()

####################################################################################################################################

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 
        
img = get_image('./archive/reduced_dataset_size/images_E_S_SB_69x69_a_03/images_E_S_SB_69x69_a_03_test/S/56157.jpg')
plt.imshow(img)
plt.savefig('image.png')
plt.close()

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf

preprocess_transform = get_preprocess_transform()

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((69,69))
    ])    

    return transf

pill_transf = get_pil_transform()

test_pred = batch_predict([pill_transf(img)])
test_pred.squeeze().argmax()

print(test_pred)
print(test_pred.squeeze().argmax())

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(pill_transf(img)), 
                                         batch_predict, # classification function
                                         top_labels=5, 
                                         hide_color=0, 
                                         num_samples=1000) # number of images that will be sent to classification function

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
img_boundry1 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry1)
plt.savefig('test1.png')
plt.close()

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
img_boundry2 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry2)
plt.savefig('test2.png')
plt.close()

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp/255.0, mask))
plt.savefig('test3.png')
plt.close()
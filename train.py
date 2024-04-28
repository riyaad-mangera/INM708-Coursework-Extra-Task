import os    
os.environ["KMP_DUPLICATE_LIB_OK"] = "True" #Kernal would repeatedly die when using matplotlib unless this line was used

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from datetime import datetime
import pickle
from model import ImageNeuralNetwork

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

train_path = r"archive\dataset\Training Data"
valid_path = r"archive\dataset\Validation Data"
test_path = r"archive\dataset\Testing Data"

train_dataset = ImageFolder(train_path, transforms.Compose([transforms.Resize((87, 87)), transforms.ToTensor()]))
valid_dataset = ImageFolder(valid_path, transforms.Compose([transforms.Resize((87, 87)), transforms.ToTensor()]))
test_dataset = ImageFolder(test_path, transforms.Compose([transforms.Resize((87, 87)), transforms.ToTensor()]))

img, label = train_dataset[0]

print(img.shape,label)
print(f'{train_dataset.classes}\n{valid_dataset.classes}\n{test_dataset.classes}')

batch_size = 64

train_batches = DataLoader(train_dataset, batch_size, shuffle = True, num_workers = 0, pin_memory = True, drop_last = True)
valid_batches = DataLoader(valid_dataset, batch_size * 2, num_workers = 0, pin_memory = True, drop_last = True)
test_batches = DataLoader(test_dataset, batch_size, shuffle = True, num_workers = 0, pin_memory = True, drop_last = True)

import matplotlib.pyplot as plt

learning_rate = 0.001
epochs = 5

model = ImageNeuralNetwork(device=device, learning_rate=learning_rate, epochs=epochs)
optimiser = torch.optim.Adam(model.parameters(), learning_rate, weight_decay = 0.0001)

model.set_optimiser(optimiser)

print(model)

model.to(device)

history = train(model, train_batches, valid_batches)

print("saving model")
with open(f'./saved_model.pkl', 'wb') as file:
        pickle.dump(model, file)

print(f'loading model')
with open(f'./saved_model.pkl', 'rb') as file:
    model = pickle.load(file)

model.to(device)

prediction = evaluate(model, test_batches)

print("test_loss: {:.4f}, test_accuracy: {:.4f}".format(prediction["valid_loss"], prediction["valid_accuracy"]))

train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

for idx in range(len(history)):
    
    train_losses.append(history[idx]["train_loss"])
    train_accuracies.append(history[idx]["train_accuracy"])
    valid_losses.append(history[idx]["valid_loss"])
    valid_accuracies.append(history[idx]["valid_accuracy"])

plt.plot(train_losses, label = "Training loss")
plt.plot(valid_losses, label = "Validation loss")
plt.title("Average loss per epoch with VGG and SGD")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon = False)
plt.savefig("loss_2.png")
plt.show()
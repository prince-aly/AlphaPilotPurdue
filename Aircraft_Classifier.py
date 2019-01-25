import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import time
import os
import copy
import torch.optim as optim
from torch.optim import lr_scheduler

# Hyperparameters
num_epochs = 5
num_classes = 3
batch_size = 5
learning_rate = 0.01

DATA_PATH = 'D:\Desktop\AlphaPilot\Aircraft_Data\Training'
MODEL_STORE_PATH = 'D:\Desktop\AlphaPilot\aircraft_models\\'

# transforms to apply to the data
trans = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

# Load Data
train_dataset = torchvision.datasets.ImageFolder(root=DATA_PATH, transform=trans)
test_dataset = torchvision.datasets.ImageFolder(root=DATA_PATH, transform=trans)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

classes = ('Commercial','Helicopter','Military')

##show random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(classes))

net = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                net.train()  # Set model to training mode
            else:
                net.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ft.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

train_model()

print('Finished Training')

torch.save(net, 'aircraft_classifier_model.pt')

##show RGB converted images
#train_loader_RGB = transforms.functional.to_grayscale(train_loader, num_output_channels=3)
#dataiter_RGB = iter(train_loader_RGB)
#images_RGB, labels_RGB = dataiter_RGB.next()
#imshow(torchvision.utils.make_grid(images))
#print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


### Convolution neural network
#class ConvNet(nn.Module):
#    def __init__(self):
#        super(ConvNet, self).__init__()
#        self.layer1 = nn.Sequential(
#            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=2, stride=2))
#        self.layer2 = nn.Sequential(
#            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=2, stride=2))
#        self.drop_out = nn.Dropout()
#        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
#        self.fc2 = nn.Linear(1000, 10)
#        
#    def forward(self, x):
#        out = self.layer1(x)
#        out = self.layer2(out)
#        out = out.reshape(out.size(0), -1)
#        out = self.drop_out(out)
#        out = self.fc1(out)
#        out = self.fc2(out)
#        return out
#    
#model = ConvNet().cuda() #use GPU
#
## Loss and optimizer
#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## Train the model
#total_step = len(train_loader)
#loss_list = []
#acc_list = []
#for epoch in range(num_epochs):
#    for i, (images, labels) in enumerate(train_loader):
#        # Run the forward pass
#        images, labels = images.cuda(), labels.cuda() #use GPU
#        outputs = model(images)
#        loss = criterion(outputs, labels)
#        loss_list.append(loss.item())
#
#        # Backprop and perform Adam optimisation
#        optimizer.zero_grad()
#        loss.backward() #trace loss to weights and improve weights
#        optimizer.step()
#
#        # Track the accuracy
#        total = labels.size(0)
#        _, predicted = torch.max(outputs.data, 1)
#        correct = (predicted == labels).sum().item()
#        acc_list.append(correct / total)
#
#        if (i + 1) % 100 == 0:
#            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
#                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
#                          (correct / total) * 100))

## Test the model
#model.eval()
#with torch.no_grad():
#    correct = 0
#    total = 0
#    for images, labels in test_loader:
#        outputs = model(images)
#        _, predicted = torch.max(outputs.data, 1)
#        total += labels.size(0)
#        correct += (predicted == labels).sum().item()
#
#    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
## we can see if validation agrees with training, within the same loop
#    
## Save the model and plot
#torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')
#
#p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch ConvNet results')
#p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
#p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
#p.line(np.arange(len(loss_list)), loss_list)
#p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
#show(p)
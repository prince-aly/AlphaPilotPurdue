from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layer2 = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.fc = nn.Linear(32 * 56 * 56, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        #print("After Layer 1", out.shape)
        out = self.layer2(out)
        #print("After Layer 2", out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        #print("Final Layer Output Shape", out.shape)
        return out

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

class FaceLandmarksSubset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, landmarks_frame, root_dir , transform=None):
      
        self.landmarks_frame = landmarks_frame
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.landmarks_frame)

def main():
    plt.ion()   #interactive mode
    
    transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                               root_dir='data/faces/',
                                               transform=transforms.Compose([
                                                   Rescale(256),
                                                   RandomCrop(224),
                                                   ToTensor()
                                               ]))
    
    dataloader = DataLoader(transformed_dataset, batch_size=3,
                            shuffle=True, num_workers=4)
    
    
    num_classes = 68 * 2 #68 coordinates X and Y flattened
    
    def random_split(dataset, training_size):
        """
        Randomly split a dataset into non-overlapping new datasets of given lengths.
    
        Arguments:
            dataset (Dataset): Dataset to be split
            
        """
    
        return (
            FaceLandmarksSubset(
                dataset.landmarks_frame[0:training_size], 
                dataset.root_dir,
                transform=transforms.Compose([
                     Rescale(256),
                     RandomCrop(224),
                     ToTensor()
                ])
            ),
            FaceLandmarksSubset(
                dataset.landmarks_frame[training_size:],
                dataset.root_dir,
                transform=transforms.Compose([
                     Rescale(256),
                     RandomCrop(224),
                     ToTensor()
                ])
            )
       )
    
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    num_epochs = 15;
    batch_size = 3;
    learning_rate = 0.001;
    data_limit = 2000
    
    n_training_samples = 51 #int(len(transformed_dataset) * 0.6)
    n_test_samples = 18 #int(len(transformed_dataset) * 0.4)
    
    train_dataset ,test_dataset = random_split(
        transformed_dataset, 
        n_training_samples
    )
    
    
    print("DATASETS",
        len(transformed_dataset), 
        len(train_dataset),
        len(test_dataset) )
    
    sample = transformed_dataset[2]
    print("FULL DATASET SAMPLE", sample['image'].shape, sample['landmarks'].shape)
    sample = train_dataset[2]
    print("TRAINING DATASET SAMPLE", sample['image'].shape, sample['landmarks'].shape)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    
    test_loader = DataLoader(test_dataset , batch_size=batch_size,
                            shuffle=True, num_workers=4)
    
    print("LOADERS",
        len(dataloader),
        len(train_loader),
        len(test_loader))

    model = ConvNet(num_classes).to(device)
#
#    # Loss and optimizer
#    criterion = nn.MSELoss()
#    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#    
#    # Train the model
#    total_step = len(train_loader)
#    for epoch in range(num_epochs):
#        for i, sample_batched in enumerate(train_loader):
#            #print(i, sample_batched['image'].size(),
#            # sample_batched['landmarks'].size())
#            
#            images_batch, landmarks_batch = \
#                sample_batched['image'], sample_batched['landmarks']
#            
#            images = images_batch
#            labels = landmarks_batch.reshape(-1, 68 * 2)
#            
#            images = Variable(images.float())
#            labels = Variable(labels)
#            
#            images = images.to(device)
#            labels = labels.to(device)
#            
#            # Forward pass
#            outputs = model(images)
#            
#            #print("Label Shape", labels.shape, "Output Shape", outputs.shape)
#            
#            
#            loss = criterion(outputs, labels.float())
#            
#            # Backward and optimize
#            optimizer.zero_grad()
#            loss.backward()
#            optimizer.step()
#            
#            if (i+1) % 5 == 0:
#                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
#
#    # Save the model checkpoint
#    torch.save(model.state_dict(), 'model.ckpt')
    
    model.load_state_dict(torch.load('model.ckpt'))
    
    def show_landmarks_batch(sample_batched):
        """Show image with landmarks for a batch of samples."""
        images_batch, landmarks_batch = \
                sample_batched['image'].cpu(), sample_batched['landmarks'].cpu()
        batch_size = len(images_batch)
        im_size = images_batch.size(2)
    
        grid = utils.make_grid(images_batch)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
    
        for i in range(batch_size):
            plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                        landmarks_batch[i, :, 1].numpy(),
                        s=10, marker='.', c='r')
    
            plt.title('Batch from dataloader')
    
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for i_batch, sample_batched in enumerate(train_loader):
            #print(i, sample_batched['image'].size(),
            # sample_batched['landmarks'].size())
            
            images_batch, landmarks_batch = \
                sample_batched['image'], sample_batched['landmarks']
            
            images = images_batch
            labels = landmarks_batch.reshape(-1, 68 * 2)
            
            images = Variable(images.float())
            labels = Variable(labels)
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            #_, predicted = torch.max(outputs.data, 1)
            #_, predicted = outputs.data
            print("Predicted", outputs.data.shape)
            
            if i_batch == 4:
              plt.figure()
              show_landmarks_batch({'image': images, 'landmarks': outputs.data.reshape(-1, 68, 2) })
              plt.axis('off')
              plt.ioff()
              plt.show()
              show_landmarks_batch({'image': images, 'landmarks': labels.reshape(-1, 68, 2) })
              plt.axis('off')
              plt.ioff()
              plt.show()
              break
            
            
            #total += labels.size(0)
            #correct += (predicted == labels).sum().item()
    
        #print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    
if __name__=='__main__':
    main()
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module
from torch.optim import SGD

#TODO
#Add in object detection for faces or masks in an image or video (this has been very difficult so far)
#Add normalization to image transforms
#Install CUDA driver
#Figure out how to save the neural network's state for later

#defining all paths using pathlib, nice library
if __name__ == '__main__':
    Data_folder = Path('C:/Users/mgwin/source/repos/mask-detection/mask-detection/mask-dataset')
    train_f = Data_folder / 'Train'
    test_f = Data_folder / 'Test'
    valid_f = Data_folder / 'Validation'
    mask_train = train_f / 'Mask'
    nomask_train = train_f / 'Non Mask'
    mask_valid = valid_f / 'Mask'
    nomask_valid = valid_f / 'Non Mask'

    #don't need many more transformations than this. Maybe a normalize? I can't figure out how to get the mean/ std dev values though
    transform = transforms.Compose([
            transforms.Resize((100,100)),
            transforms.ToTensor()
        ])

    #define datasets: imagefolder seemed like the simplest solution and has had the fastest execution time
    trainDataset = ImageFolder(train_f, transform=transform)
    testDataset = ImageFolder(test_f, transform=transform)

    #define dataloaders: test images have a loader batch size of 10 so that I can display more by taking one slice at the end
    trainDataLoader = DataLoader(trainDataset, batch_size=4, shuffle=True, num_workers=1)
    testDataLoader = DataLoader(testDataset, batch_size=10, shuffle=True, num_workers=1)

    #Get device for training: I don't want to install the CUDA driver, so this will most likely always be cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    #Define neural network class: layers were created based off another image classifier network for detecting expressions
    class NeuralNetwork(Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.cnn_layers = Sequential(
                # Defining a 2D convolution layer
                Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=(2, 2)),
                # Defining another 2D convolution layer
                Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=(2, 2)),
                #Defining third convolution layer
                Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(3,3)),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=(2, 2)),
            )

            self.linear_layers = Sequential(
                Linear(in_features=2048, out_features=1024),
                ReLU(),
                Linear(in_features=1024, out_features=2),
            )

        # Defining the forward pass    
        def forward(self, x):
            x = self.cnn_layers(x)
            x = x.view(x.size(0), -1)
            x = self.linear_layers(x)
            return x

    model = NeuralNetwork()
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    #iterate through dataloader to pass data to network: mostly from CIFAR10 tutorial documentation
    for epoch in range(16):
        running_loss = 0.0
        for (idx, batch) in enumerate(trainDataLoader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if idx % 50 == 49:    # print every 50 mini-batches
                print('[%d, %5d] loss: %.4f' %
                        (epoch + 1, idx + 1, running_loss / 50))
                running_loss = 0.0

    print('Finished Training')

    #iterate through dataloader for testing
    dataiter = iter(testDataLoader)
    images, labels = dataiter.next()

    #print images for testing
    plt.imshow(utils.make_grid(images, nrow=10).permute(1, 2, 0))
    print('Ground Truth: ', ' '.join('%5s |' % testDataset.classes[labels[j]] for j in range(10)))

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s |' % testDataset.classes[predicted[j]] for j in range(10)))

    #calculate overall network accuracy for all test images
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data in testDataLoader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

    plt.show()
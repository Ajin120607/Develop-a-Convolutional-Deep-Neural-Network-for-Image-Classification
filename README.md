# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
he problem is to design and develop a Convolutional Deep Neural Network (CNN) that can automatically classify grayscale images into predefined categories. The model must learn important spatial features such as edges, textures, and shapes from image data and accurately predict the correct class label.

## Neural Network Model
<img width="1024" height="751" alt="image" src="https://github.com/user-attachments/assets/93bdbae8-0b65-4055-b259-4f5163e8cdfb" />

## DESIGN STEPS
### STEP 1: 
Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.

### STEP 2: 
Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.

### STEP 3: 
Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

### STEP 4: 
Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.

### STEP 5: 
Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.

### STEP 6: 
Evaluate the trained model on test images and verify the classification accuracy for new unseen images.

## PROGRAM

### Name: AJIN A

### Register Number: 212224230011

```python

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # write your code here
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)




    def forward(self, x):
        # write your code here
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)
## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):

    # write your code here
    for epoch in range(num_epochs):
        model.train()
        running_loss=0.0
        for images,labels in train_loader:
          optimizer.zero_grad()
          outputs=model(images)
          loss=criterion(outputs,labels)
          loss.backward()
          optimizer.step()
          running_loss+=loss.item()


        print('Name: AJIN A')
        print('Register Number: 212224230011')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

### OUTPUT

## Training Loss per Epoch

<img width="296" height="194" alt="image" src="https://github.com/user-attachments/assets/16a94044-fefe-43dd-942f-d3f2b1a28ab6" />


## Confusion Matrix

<img width="750" height="695" alt="image" src="https://github.com/user-attachments/assets/e1dfe9da-1f09-43fc-ba8e-ae7eb5ac753f" />


## Classification Report
<img width="440" height="345" alt="image" src="https://github.com/user-attachments/assets/3e2768c2-5558-4a8e-98d7-0d8468c62a33" />


### New Sample Data Prediction
<img width="416" height="487" alt="image" src="https://github.com/user-attachments/assets/d9738083-9ca3-4f85-9bb8-6e51a49eb32e" />


## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.

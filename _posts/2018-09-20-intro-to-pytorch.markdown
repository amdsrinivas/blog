---
layout: post
title: "Introduction to PyTorch"
date: 2018-09-20
description: Introduction to Data processing pipeline using PyTorch! # Add post description (optional)
img:  intro_to_pytorch/pytorch-logo-flat.png # Add image post (optional)
url: blog/intro-to-pytorch
---

Pytorch Basics
====
[Pytorch](pytorch.org) is a relatively newer deep learning framework compared to Tensorflow. But, it provides excellent features that makes it a framework you should checkout. Main features are **Dynamic Graphs and Typing**.
In this article, we will deal with Pytorch's Dataset class to load, transform and create minibatches for training neural networks in Pytorch. I will explain this by training a simple regression model.
---
First, we will need a simple regression data that would not overwhelm basic networks, yet would need some effort to fit. For this purpose, we will use [Sklearn's make_regression](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html) function. The _make_regression()_ function generates multi-dimensional regressable data using a linear model. We can add noise(Guassian) to this data to increase our network's work to fit this data. The following code demonstrates the use of this method.

```python
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

#Generate the data.
X, y = make_regression(n_samples=500, n_features=1, noise=25.0)

# Visualise the data.
plt.scatter(X,y)
plt.show()
```
This data is generated based on a random linear model due to which, it is different for each time it is generated. In my case, the generated data is shown below:
![Data image]({{site.baseurl}}/assets/img/intro_to_pytorch/data.png)
---
Once the data is ready, we have to work on data processing and data loading. This is where the Pytorch's Dataset class shines. All we need to do is to inheirt the Dataset class and override the __init, len and getitem__ methods.

In the \_\_init\_\_() method, we load the data required and set the class parameters. The following code demonstrates the \_\_init\_\_() that I used. This dataset is too basic to actually appreciate the use of _transform_ parameter. I will explain it in a future post.
```python
def __init__(self, num_samples=500, num_features=1, num_targets=1, Noise=25.0, transform=None):
    self.X , self.Y = make_regression(n_samples=num_samples, n_features=num_features, noise=Noise)

    #Normalize the data across the dataset.
    X_min, X_max = np.min(self.X), np.max(self.X)
    self.X = np.array(list(map(lambda k: (k - X_min)/(X_max-X_min) , self.X) ))
    Y_min, Y_max = np.min(self.Y), np.max(self.Y)
    self.Y = np.array(list(map(lambda k: (k-Y_min)/(Y_max-Y_min), self.Y)))

    self.transform = transform
```

The \_\_len\_\_() method should simply return the number of data samples available.
```python
def __len__(self):
    return len(self.X)
```

The \_\_getitem\_\_() should return the data point at any given _idx_.
```python
def __getitem__(self, idx):

    sample = {'X': self.X[idx], 'Y':self.Y[idx]}

    if self.transform :
        sample = self.transform(sample)
        
    return sample
```
The total Dataset class would be as shown:
```python
class RegressionData(Dataset):
    def __init__(self, num_samples=500, num_features=1, num_targets=1, Noise=25.0, transform=None):
        self.X , self.Y = make_regression(n_samples=num_samples, n_features=num_features, noise=Noise)
        
        #Normalize the data across the dataset.
        X_min, X_max = np.min(self.X), np.max(self.X)
        self.X = np.array(list(map(lambda k: (k - X_min)/(X_max-X_min) , self.X) ))
        Y_min, Y_max = np.min(self.Y), np.max(self.Y)
        self.Y = np.array(list(map(lambda k: (k-Y_min)/(Y_max-Y_min), self.Y)))
        
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):

        sample = {'X': self.X[idx], 'Y':self.Y[idx]}

        if self.transform :
            sample = self.transform(sample)
        
        return sample
```
Now that we have our dataset ready, we can use DataLoader object to load batches from it. It is demonstrated below:
```python
dataset = RegressionData()
dl = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
batch_iterator = iter(dl)
for i in range(2):
    print(batch_iterator.next())
```
The above code implements a data loader that outputs a randomly shuffled 8 sample batches. The output of the above code is given below:
```output
{'X': tensor([[-1.0382],
        [-1.1148],
        [ 0.7841],
        [ 0.5795],
        [-0.2476],
        [-0.6030],
        [ 0.8778],
        [-0.6523]], dtype=torch.float64), 
        'Y': tensor([-51.3919, -11.6604,  55.9812, -28.9382, -41.9735,  19.2593,  21.8576, -57.9561], dtype=torch.float64)}
{'X': tensor([[ 0.6411],
        [-0.1372],
        [-0.2143],
        [ 0.5077],
        [ 0.4351],
        [ 0.8013],
        [-0.1157],
        [ 0.0337]], dtype=torch.float64), 'Y': tensor([ 42.2070,  15.3320,  53.6430,  18.9361,  30.9218,  21.3723, -22.0434, 24.7658], dtype=torch.float64)}
```
---
Now that the data is ready, let us start building a feed forward neural network. Let us use a simple network with a __single hidden layer with 4 nodes__ with ReLU activation. For this purpose, we can use the _nn.Module_ class of pytorch and implement the constructor and _forward()_ methods. The follwoing code demonstrates this:
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()

        self.layer1 = nn.Linear(1,4)
        self.layer2 = nn.Linear(4,1)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        return x
```
---
Now that we have the data and model ready, we can train the model. Firstly, we need to create a dataset, dataloader, optimizer and the Neural network classes as shown below.ALonf with these, we need to specify on what metric the network is to be optimized. Since, we are dealing with a Regression problem, we can use MSE as the loss criterion.
```python
dataset = RegressionData()
dl = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()
model = NeuralNet()
model.double()
```
We can have a desired number of epochs and train the model using mini-batches of size 8. On every batch, the optimizer has to be stepped forward and the loss obtained should be back propagated. This is shown below:
```python
for i in range(50):

    for batch in dl:
        optimizer.zero_grad()   # zero the gradient buffers
        output = model(batch['X'])
        loss = criterion(output, batch['Y'].unsqueeze(-1))
        #print(loss)
        loss.backward()
        optimizer.step()
    print(loss)
```

Once the model is trained, we can save it as shown below:
```python
save(model.state_dict(), 'intro_to_pytorch')
```
Predictions from the model can be done through simple steps:
```python
model = NeuralNet()
model.load_state_dict(load('./intro_to_pytorch'))
predictions = model(Input_data) # We will get normalized outputs.
```
---
In summary, we have implemented a regression model using Pytorch _Dataset_ and _Module_ classes to minimize the MSE on a random dataset created through sklearn. The code used is available on [github](https://github.com/amdsrinivas/Blog-Codes). Feel free to share your feedback and see you in the next post. Stay tuned!
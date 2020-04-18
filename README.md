# MiniMLCore

Welcome to the MiniMLCore project. This project is meant to be a small and simple machine learning framework. 
## Installation
To install the MiniMLCore type teh command below into command prompt
```
pip install minimlcore
```


## Getting started
To get started create a model like so
```
    from MiniMLCore.Model import SequentialModel
    MyModel = SequentialModel(1)
```

From there add the layers that you want to add using
the model.add function. Add called layers from the layers file

```
  from MiniMLCore.Layers import *
  #MyModel.add(Example Layer Here)
  MyModel.add(Dense(150))
  MyModel.add(Relu())
  MyModel.add(Dense(1))
  MyModel.add(Sigmoid())
  MyModel.build()
```
The final Model.build at the end converts the model from an uncompiled
version to a compiled one that can be trained and can predict values, however
any changes to the model now (such as calling model.add) will break it

Now lets generate some sample data for the model to fit to (I will do a sin function) and plot it
```
import matplotlib.pyplot as plt
inputs = np.arange(0,3.14,.01)
outputs = np.sin(np.abs(np.arange(0,3.14,.01)))
plt.plot(inputs,outputs)
```

We can take this data and train our ML model from earlier on it like by doing a few things:
First we must specify an optimizer (Adam or SGD as of now):
```
from MiniMLCore.Optimizer import Adam,MeanSquaredError
#Setup like so:
#Optimizer = Adam(modelname,loss *must be one imported from Optimizer*)
Optimizer = Adam(Model,MeanSquaredError)
```
Now we can train the model
```
Optimizer.train(inputs,outputs,epochs=1000)
```
And make predictions with it too
```
Model.batch_predict(np.arange(0.0,3.14,.01))
```


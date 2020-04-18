from Model import SequentialModel
import Layers
import Optimizer
import numpy as np
import matplotlib.pyplot as plt

Model = SequentialModel(1)
Model.add(Layers.Dense(150))
Model.add(Layers.Relu())
Model.add(Layers.Dense(1))
Model.add(Layers.Sigmoid())
Model.build()
# In[]
#Prediction_test = Model.predict([1,0])
Gradient_calculations = Optimizer.Adam(Model,Optimizer.MeanSquaredError,learning_rate=.01,batch_size=999)
gradients = Gradient_calculations.gradient_calc([1],[0])

# In[]
inputs = np.arange(0,3.14,.01)
outputs = np.sin(np.abs(np.arange(0,3.14,.01)))
plt.plot(inputs,outputs)
plt.plot(np.arange(0.0,3.14,.01),np.array([Model.predict(i) for i in np.arange(0.0,3.14,.01)]).reshape(-1))

# In[]
A = Gradient_calculations.train(inputs,outputs,epochs=200)
# In[]
plt.plot(np.arange(0.0,10,.01),np.array([Model.predict(i) for i in np.arange(0.0,10,.01)]).reshape(-1))


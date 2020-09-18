import numpy as np
import matplotlib.pyplot as plt
from MiniMLCore.Model import SequentialModel
from MiniMLCore.Layers import Dense,Relu,Sigmoid,CudaDense
from MiniMLCore.Optimizer import Adam,MeanSquaredError



Model = SequentialModel(1)

Model.add(CudaDense(150))
Model.add(Relu())
Model.add(Dense(1))
Model.add(Sigmoid())
Model.build()
# In[]
#Prediction_test = Model.predict([1,0])
Gradient_calculations = Adam(Model,MeanSquaredError,learning_rate=.001,batch_size=999)
gradients = Gradient_calculations.gradient_calc([1],[0])

# In[]
inputs = np.arange(0,3.14,.01)
outputs = np.sin(np.abs(np.arange(0,3.14,.01)))
plt.plot(inputs,outputs)
plt.plot(np.arange(0.0,3.14,.01),Model.batch_predict(np.arange(0.0,3.14,.01)))

# In[]
A = Gradient_calculations.train(inputs,outputs,epochs=1000)
 # In[]
plt.plot(np.arange(0.0,10,.01),np.array([Model.predict(i) for i in np.arange(0.0,10,.01)]).reshape(-1))


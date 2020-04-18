from MiniMLCore import Model
import numpy as np
from itertools import islice 

class Losses():
    "A Generala class for losses"
    def __init__():
        pass


class MeanSquaredError(Losses):
    "The mean squared error loss function"
    def calculate_loss(expected,outputs):
        """
        A function to calculate the model loss

        Parameters
        ----------
        expected : np.array
            A numpy array for the expected values
        outputs : np.array
            A numpy array for the actual value

        Returns
        -------
        float
            The loss

        """
        return np.average((expected-outputs)**2)
    def calculate_gradients(expected,outputs):
        """
        A function to calculate the gradient of the loss given inputs and expected  balues

        Parameters
        ----------
        expected : np.array
            expected values
        outputs : np.array
            actual values

        Returns
        -------
        np.array
            Loss gradients

        """
        return 2*(expected-outputs)


class GradientOptimizer():
    """A class to calculate the gradients needed for the optimizers to compute the best course of action"""
    def __init__(self,model,cost):
        """
        The constructor for the gradient optimizer
        
        Parameters
        ----------
        model : Model
            The model that will be optimized
        cost : Losses
            A Loss function

        Returns
        -------
        None.

        """
        self.model = model
        self.cost = cost
    def gradient_calc(self,inputs,outputs):
        """
        This function calculates the gradients of the model

        Parameters
        ----------
        inputs : np.array
            model inputs
        outputs : np.array
            model outputs

        Returns
        -------
        np.array
            Gradients

        """
        inputs = np.array(inputs)
        outputs = np.array(outputs)   
        results = self.model.getlayerbylayer(inputs)
        cost_delta =  np.array(self.cost.calculate_gradients(outputs,results[-1])).reshape(-1)
        gradients = [{"PrevLayer":np.array(cost_delta)}]
        for i in range(len(self.model.layers)-1,-1,-1): 
            gradients.append(self.model.layers[i].calculate_gradients(results[i],cost_delta))
            cost_delta = np.average(np.array(gradients[-1]["PrevLayer"]),0).reshape(-1)
        return gradients[::-1]
    def batch_gradient_calculate(self,inputs,outputs):
        """
        Calculates the gradients in batches

        Parameters
        ----------
        inputs : np.array
            batch of inputs
        outputs : np.array
            batch of outputs

        Returns
        -------
        gradients : np.array
            Gradients of model

        """
        gradients = []
        for i in zip(inputs,outputs):
            gradients.append(self.gradient_calc(i[0],i[1]))
        return gradients
    def generate_batches(self,inputs,outputs):
        """
        A function to generate batches from inputs and outputs
        with the given batch_size

        Parameters
        ----------
        inputs : np.array
            inputs to the model
        outputs : np.array
            expected outputs from the model

        Returns
        -------
        None.

        """
        length = len(inputs)
        split_sizes = []
        for i in range(0,int(np.floor(length/self.batch_size))):
            split_sizes.append(self.batch_size)
        split_sizes.append(length%self.batch_size)
        return([list(islice(inputs, elem)) for elem in split_sizes],[list(islice(outputs, elem)) for elem in split_sizes])

class SGD(GradientOptimizer):
    """The Sochastic Gradient Descent Optimizer model"""
    def __init__(self,model,cost,learning_rate=.01,batch_size=32):
        """
        

        Parameters
        ----------
        model : Model
            The model that you want to train
        cost : Losses
            The loss function to use
        learning_rate : float, optional
            The learning rate of the model. The default is .01.
        batch_size : int, optional
            The batch size to subdivide the data into. The default is 32.

        Returns
        -------
        None.

        """
        super().__init__(model,cost)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
    def combine_dicts(self,a, b):
        """
        
        Combinging two dicts together used to add and average the gradients
        
        Parameters
        ----------
        a : dict
            The first dict
        b : dict
            The second dict

        Returns
        -------
        dict
            The sum of the two

        """
        return dict([(k, np.add(a[k], b[k])) for k in set(b) & set(a)])
    def train_step(self,inputs,outputs):
        """
        A training step iterated over for an epoch

        Parameters
        ----------
        inputs : np.array
            models inputs
        outputs : np.array
            models outputs

        Returns
        -------
        float
            A sample gradient from teh Model 

        """
        batches_in,batches_out = self.generate_batches(inputs,outputs)
        for idx in range(len(batches_in)):
            gradients = self.batch_gradient_calculate(batches_in[idx],batches_out[idx])
            average_gradients = gradients[0]
            for gradient in gradients:
                for layerindex in range(0,len(gradient)):
                    layer = gradient[layerindex]
                    avg_grads_layer = average_gradients[layerindex]
                    average_gradients[layerindex] = self.combine_dicts(layer,avg_grads_layer)
            for entry_num in range(0,len(average_gradients)):
                for key in average_gradients[entry_num]:
                    average_gradients[entry_num][key] /= (len(gradients)/self.learning_rate)
            self.model.apply_changes(average_gradients)
        return average_gradients[-1]["PrevLayer"][-1]
    def train(self,inputs,outputs,epochs=1):
        """
        The training methods

        Parameters
        ----------
        inputs : np.array
            models inputs
        outputs : np.array
            models outputs
        epochs : int, optional
            number of times to iterate the training step. The default is 1.

        Returns
        -------
        loss_list : float
            the loss of the model

        """
        
        loss_list = []
        for index in range(0,epochs):
            self.train_step(inputs,outputs)
            loss = self.cost.calculate_loss(self.model.batch_predict(inputs),outputs)
            loss_list.append(loss)
            print(loss)
        return loss_list
    
class Adam(GradientOptimizer):
    """The Adam optimizer for models"""
    def __init__(self,model,cost,learning_rate=.001,batch_size=32,beta_1=.9,beta_2=.999,epsilon=1e-8):
        """
        

        Parameters
        ----------
        model : Model
            The model needed to optimize
        cost : Losses
            The cost function that needs to be used
        learning_rate : float, optional
            The leraning rate for the adam optimizer. The default is .001.
        batch_size : int, optional
            The Batch Size. The default is 32.
        beta_1 : float, optional
            The beta value for the first order. The default is .9.
        beta_2 : float, optional
            The beta value used for the second order. The default is .999.
        epsilon : float, optional
            The epsilon value used in adam optimization. The default is 1e-8.

        Returns
        -------
        None.

        """
        super().__init__(model,cost)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
        self.v_momentum = False
        self.s_momentum = False
        self.t = 0
    def combine_dicts(self,a, b):
        """
        
        Combinging two dicts together used to add and average the gradients
        
        Parameters
        ----------
        a : dict
            The first dict
        b : dict
            The second dict

        Returns
        -------
        dict
            The sum of the two

        """
        return dict([(k, np.add(a[k], b[k])) for k in set(b) & set(a)])
    
    def train_step(self,inputs,outputs):
        """
        The training methods

        Parameters
        ----------
        inputs : np.array
            models inputs
        outputs : np.array
            models outputs
        epochs : int, optional
            number of times to iterate the training step. The default is 1.

        Returns
        -------
        loss_list : float
            the loss of the model

        """
        
        batches_in,batches_out = self.generate_batches(inputs,outputs)
        self.t += 1
        for idx in range(len(batches_in)):
            gradients = self.batch_gradient_calculate(batches_in[idx],batches_out[idx])
            average_gradients = gradients[0]
            for gradient in gradients:
                for layerindex in range(0,len(gradient)):
                    layer = gradient[layerindex]
                    avg_grads_layer = average_gradients[layerindex]
                    average_gradients[layerindex] = self.combine_dicts(layer,avg_grads_layer)
            for entry_num in range(0,len(average_gradients)):
                for key in average_gradients[entry_num]:
                    average_gradients[entry_num][key] /= -len(gradients)
                    
            if self.v_momentum == False:  
                self.v_momentum = average_gradients.copy()
                for entry_num in range(0,len(average_gradients)):
                    for key in average_gradients[entry_num]:
                        self.v_momentum[entry_num][key] = np.zeros(np.shape(self.v_momentum[entry_num][key]))
                              
            if self.s_momentum == False:                     
                self.s_momentum = average_gradients.copy()
                for entry_num in range(0,len(average_gradients)):
                    for key in average_gradients[entry_num]:
                        self.s_momentum[entry_num][key] = np.zeros(np.shape(self.s_momentum[entry_num][key]))
            
            for entry_num in range(0,len(average_gradients)):
                for key in average_gradients[entry_num]:
                    
                    self.v_momentum[entry_num][key] = self.beta_1*self.v_momentum[entry_num][key] + (1-self.beta_1)*average_gradients[entry_num][key]
                    self.s_momentum[entry_num][key] = np.abs(self.beta_2*self.s_momentum[entry_num][key]) + (1-self.beta_2)*((average_gradients[entry_num][key])**2)
                    
                    v_adj = self.v_momentum[entry_num][key]/(1-(self.beta_1**self.t))
                    s_adj = self.s_momentum[entry_num][key]/(1-(self.beta_2**self.t))
                    
                    opt_val = (v_adj/((np.sqrt(s_adj)+self.epsilon)))
                    average_gradients[entry_num][key] = -self.learning_rate*opt_val

            self.model.apply_changes(average_gradients)
        return average_gradients[-1]["PrevLayer"][-1]
    def train(self,inputs,outputs,epochs=1):
        """
        The training methods

        Parameters
        ----------
        inputs : np.array
            models inputs
        outputs : np.array
            models outputs
        epochs : int, optional
            number of times to iterate the training step. The default is 1.

        Returns
        -------
        loss_list : float
            the loss of the model

        """
        
        loss_list = []
        for index in range(0,epochs):
            self.train_step(inputs,outputs)
            loss = self.cost.calculate_loss(self.model.batch_predict(inputs),outputs)
            loss_list.append(loss)
            print(loss)
        return loss_list

import numpy as np


class Neuron():
    """
    A basic neuron class
    """
    def __init__(self,input_shape):
        """
        The basic initaliser for a neurohn

        Parameters
        ----------
        input_shape : int
            the input shape

        Returns
        -------
        None.

        """
        self.input_shape = input_shape

class SimpleNeuron(Neuron):
    """
    The class used in the dense layers
    """
    def __init__(self,input_shape):
        """
        The constructor the neuron 

        Parameters
        ----------
        input_shape : int
            the input shape

        Returns
        -------
        None.

        """
        super().__init__(input_shape)
        self.weights = np.random.uniform(-1,1,size=input_shape)
        self.bais = [np.random.normal()]
        self.input_compute = self.lastlayercompute
    def compute(self,inputs):
        """
        Computes the outputs given an input

        Parameters
        ----------
        inputs : np.array
            the inputs

        Returns
        -------
        Tnp.array
            outputs of the neurons results

        """
        return np.sum(self.weights*inputs)+self.bais[0]
    def weightcompute(self,inputs,delta):
        #Computes the partial derivative of the input with respect to the cost
        return inputs*delta
    def baiscompute(self,inputs,delta):
        #Computes the partial derivative of the bais with respect to the cost
        return delta
    def lastlayercompute(self,inputs,delta):
        #Computes the partial derivative of the lastlayer's outputs with respect to the cost
        return self.weights*delta
    
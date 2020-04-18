import Neurons
import numpy as np


"""
Todo: remake some of these such as the activations using metaclasses because they are very simular with only changes to compute and derivative

"""







class Layer():
    """
    Layer Basic Class
    """
    def __init__(self):
        pass



class Dense(Layer):
    """Here we have a basic densely connected neuron layer
    It calls the SimpleNeuron cells as to function as Neurons
    """
    
    def __init__(self,units):
        """
        Generates the neurons

        Parameters
        ----------
        units : int
            The number of neurons to generate in the layer

        Returns
        -------
        None.

        """
        super().__init__()
        self.units = units
        self.cells = [Neurons.SimpleNeuron for i in range(0,self.units)]
    def build(self,input_shape):
        """
        Builds the layerand the neurons so that they can process inputs   
        
        Parameters
        ----------
        input_shape : int
            The input shape of the neurons input

        Returns 
        -------
        int
            the number of neurons

        """
        temp_cells = []
        for cell in self.cells:
            temp_cells.append(cell(input_shape))
        self.cells = temp_cells
        return self.units
    def compute(self,inputs):
        """
        Computes the results of the layer

        Parameters
        ----------
        inputs : np.array
            inputs (or outputs from last layer) works on one sample

        Returns
        -------
        np.array
             The results of that computation on the output

        """
        return np.array([cell.compute(inputs) for cell in self.cells])
    def calculate_gradients(self,inputs,deltas):
        """
        Calculates a few partial derivatives for the layer, used in
        the optimizers, namley the current gradient of the weights baises and the previous layer\'s weights to the
        cost function
            

        Parameters
        ----------
        inputs : np.array
            the layers inputs
        deltas : int
            the previous deltas for the layers that came before

        Returns
        -------
        part_gradients : dict
            The partial derivcatives for the weights baises and the previous activation layer with respect to the cost function

        """
        part_gradients = dict()
        part_gradients["Weights"] = [self.cells[idx].weightcompute(inputs,deltas[idx]) for idx in range(0,len(self.cells))]
        part_gradients["Baises"] = [self.cells[idx].baiscompute(inputs,deltas[idx]) for idx in range(0,len(self.cells))]
        part_gradients["PrevLayer"] = [self.cells[idx].lastlayercompute(inputs,deltas[idx]) for idx in range(0,len(self.cells))]
        return part_gradients
    def apply_changes(self,inputs):
        for idx in range(0,len(inputs["Weights"])):
            self.cells[idx].weights = np.add(self.cells[idx].weights,inputs["Weights"][idx])
        for idx in range(0,len(inputs["Baises"])):
            self.cells[idx].bais = np.add(self.cells[idx].bais,inputs["Baises"][idx])
        
        
        
class Sigmoid(Layer):
    """
    This is a class for the sigmoid activation layer
    """
    def __init__(self):
        """
        Initializes the layer

        Returns
        -------
        None.

        """
        super().__init__()
        pass
    def build(self,input_shape):
        """
        Builds the layer with the input shape and returns the input shape

        Parameters
        ----------
        input_shape : int
            The input shape that the activation will recive

        Returns
        -------
        input_shape : int
            The output shape that the activation will give (same as the input shape)

        """
        self.input_shape = input_shape
        return input_shape
    def compute(self,inputs):
        """
        Computes the results of this layer

        Parameters
        ----------
        inputs : double,int,np.array,ect
            applies the sigmoid function to the inputs

        Returns
        -------
        float,np.array
            the output of the sigmoid function

        """
        return (1/(1+(np.e**-inputs)))
    def derivative(self,inputs):
        """
        The Derivative of the sigmoid function, used to calculate some costs

        Parameters
        ----------
        inputs : double,int,np.array,ect
            inputs

        Returns
        -------
        np.array,double
            The derivative of the sigmoid functions inputs wrt the output

        """
        return self.compute(inputs)*(1-self.compute(inputs))
    def calculate_gradients(self,inputs,deltas):
        """
        A function to calculate partial derivates to use in gradient calculations
        calculate the gradient of inputs with respect to the cost function

        Parameters
        ----------
        inputs : np.array
            Inputs to the klayer
        deltas : np.array
            The deltas of previous layers in the path
            
        Returns
        -------
        part_gradients : TYPE
            gradient of inputs with respect to the cost function

        """
        part_gradients = dict()
        part_gradients["PrevLayer"] = np.array(self.derivative(inputs)*deltas).reshape(1,-1)
        return part_gradients
    def apply_changes(self,inputs):
        """
        Does nothing because there are no trainable weights for an activation
        Parameters
        ----------
        inputs : np.array
            inputs

        Returns
        -------
        None.

        """
        pass
    
class Relu(Layer):
    """A Relu activation layer"""
    def __init__(self):
        """
        Generates a Relu Layer

        Returns
        -------
        None.

        """
        super().__init__()
        pass
    def build(self,input_shape):
        """
        Builds the layer to be used

        Parameters
        ----------
        input_shape : int
            the input shape

        Returns
        -------
        input_shape : int
            the output shape(same as input_shape)

        """
        self.input_shape = input_shape
        return input_shape
    def compute(self,inputs):
        """
        Computes the outputs of calling the layer

        Parameters
        ----------
        inputs : np.array
            the inputs

        Returns
        -------
        np.array
            The outputs of relu

        """
        return np.maximum(0,inputs)
    def derivative(self,inputs):
        """
        The derivate of the Relu Function

        Parameters
        ----------
        inputs : np.array
            the inputs to the layer

        Returns
        -------
        np.array
            the outputs of the derivaive of relu

        """
        return np.clip(np.ceil(inputs),0,1)
    def calculate_gradients(self,inputs,deltas):
        """
        Calculates the gradient of the inputs with respect ot the cost function

        Parameters
        ----------
        inputs : np.array
            layer inputs
        deltas : np.array
            last layer deltas

        Returns
        -------
        part_gradients : np.array()
            the partial derivative of the inputs with respect to the change in network cost

        """
        part_gradients = dict()
        part_gradients["PrevLayer"] = np.array(self.derivative(inputs)*deltas).reshape(1,-1)
        return part_gradients
    def apply_changes(self,inputs):
        """
        No trainable weights so function does nothing

        Parameters
        ----------
        inputs : np.array
            inputs

        Returns
        -------
        None.

        """
        pass

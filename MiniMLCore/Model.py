import numpy as np


class SequentialModel():
    """ 
    This is the model class to generate a neural netowrk
    to get started with this use the type in SequentialMode(inputs)
    and then substitute inputs for the input shape (1-D) of the data 
    that will be inputted
    
    from there add the layers you want (I.E. Dense, Activation Layers) 
    and once you are finished call the build() command to convert the
    model from a skeleton model into a compiled model
    
    Now that the model has been compiled you can use predict to predict
    for 1 input or batch predict to simultanously predict multiple
    
    To train your model so it is useful first compilie it and then use
    One of the optimizers (Adam is the better one) to train it: For full
    documentation on this read the docstring in Optimizer.py
    
    """

    def __init__(self,input_shape:int):
        """
        Parameters
        ----------
        input_shape : int
            Takes the input shape and generates a model with it

        Returns
        -------
        None.

        """
        self.input_shape = input_shape
        self.build_shapes = [input_shape]
        self.layers = []
        
    def add(self,layer_to_add):
        """
        Parameters
        ----------
        layer_to_add : Layer
            Takes a layer and adds it to the network

        Returns
        -------
        None.

        """
        self.layers.append(layer_to_add)
        
    def build(self):
        """
        Takes the model and builds it and compiles it

        Returns
        -------
        None.

        """
        for layer in self.layers:
            self.build_shapes.append(layer.build(self.build_shapes[-1]))
    def predict(self,inputs):
        """
        Parameters
        ----------
        inputs : list
            single input to the model

        Returns
        -------
        inputs : list
            output for that input

        """
        inputs = np.array(inputs)
        for layer in self.layers:
            inputs = layer.compute(inputs)
        return inputs
    def batch_predict(self,inputs):
        """
        Parameters
        ----------
        inputs : list
            list of inputs to the model

        Returns
        -------
        inputs : list
            list of outputs for that input

        """
        outputs = []
        for single_input in inputs:
            outputs.append(self.predict(single_input))
        return outputs
    def getlayerbylayer(self,inputs):
        """
        This function is used mainly to calculate the gradients

        Parameters
        ----------
        inputs : list
            single input to the model

        Returns
        -------
        z_vectors : list
            results of each layer in the model

        """
        inputs = np.array(inputs)
        z_vectors = [inputs]
        for layer in self.layers:
            inputs = layer.compute(inputs)
            z_vectors.append(inputs)
        return z_vectors
    def apply_changes(self,changes):
        """
        Applys the changes given back from the optimizer to the model

        Parameters
        ----------
        changes : list
            Changes list from the optimizers

        Returns
        -------
        None.

        """
        for index in range(0,len(self.layers)):
            self.layers[index].apply_changes(changes[index])
    
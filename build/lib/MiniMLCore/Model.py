from MiniMLCore.Layers import Layer
from MiniMLCore.OperationsManagers.CPU import CPU
class Sequential():
    def __init__(self,input_shape :list,operations=CPU):
        self.Network = list()
        self.input_shape = list(input_shape)
        self.operations = operations
    def add(self,layer :Layer):
        self.Network.append(layer)
    def build(self):
        temp_shape = self.input_shape
        for layer in self.Network:
            layer.build(temp_shape,operations=self.operations)
            temp_shape = layer.get_output_shape()
        self.output_shape = temp_shape
    def predict(self,inputs):
        for layer in self.Network:
            inputs = layer.compute(inputs)
        return inputs
    def batchPredict(self,inputs):
        inputs = self.operations.default_array(inputs)
        outputs = []
        for sub in inputs:
            outputs.append(self.predict(sub))
        return self.operations.default_array(outputs)
    def layerByLayerPredict(self,inputs):
        inputs = [inputs]
        for layer in self.Network:
            inputs.append(layer.compute(inputs[-1]))
        return inputs
    def applyChanges(self,changes):
        for layer in zip(self.Network,changes):
            layer[0].apply_changes(layer[1])

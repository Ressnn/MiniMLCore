from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def computeLoss(expected,outputs):
        return float
    @abstractmethod
    def computeGradients(expected,outputs):
        return float

class MeanSquaredError(Loss):
    def computeLoss(expected,outputs):
        return (expected-outputs)**2
    def computeGradients(expected,outputs):
        return 2*(expected-outputs)


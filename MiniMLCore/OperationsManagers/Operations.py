from abc import ABC, abstractmethod

class OperationsManager(ABC):
    where = None;
    @abstractmethod
    def dot(a,b):
        pass
    @abstractmethod
    def add(a,b):
        pass
    @abstractmethod
    def subtract(a,b):
        pass
    @abstractmethod
    def divide(a,b):
        pass
    @abstractmethod
    def multiply(a,b):
        pass
    @abstractmethod
    def shape(shape):
        pass
    @abstractmethod
    def average(arr,layer=0):
        pass
    @abstractmethod
    def matmul(arr1,arr2):
        pass
    @abstractmethod
    def reshape(arr,shape):
        pass
    @abstractmethod
    def default_array():
        pass
    @abstractmethod
    def get_random_method():
        pass
    

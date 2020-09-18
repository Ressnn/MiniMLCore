import numpy as np
from MiniMLCore.OperationsManagers.Operations import OperationsManager

class CPU(OperationsManager):
    where = np.where
    def dot(a,b):
        return np.dot(a,b)
    def add(a,b):
        return np.add(a,b)
    def subtract(a,b):
        return np.subtract(a,b)
    def divide(a,b):
        return np.divide(a,b)
    def multiply(a,b):
        return np.multiply(a,b)
    def zeroes(shape):
        return np.zeros(shape)
    def average(arr,layer=0):
        return np.average(arr,layer)
    def matmul(arr1,arr2):
        return np.matmul(arr1,arr2)
    def reshape(arr,shape):
        return np.reshape(arr, shape)
    def default_array(inputs):
        return np.array(inputs)
    def get_random_method():
        return np.random.normal



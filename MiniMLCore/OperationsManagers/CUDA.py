import cupy
from MiniMLCore.OperationsManagers.Operations import OperationsManager

class CUDAAcceleratedOperations(OperationsManager):
    where=cupy.where
    def dot(a,b):
        return cupy.dot(a,b)
    def add(a,b):
        return cupy.add(a,b)
    def subtract(a,b):
        return cupy.subtract(a,b)
    def divide(a,b):
        return cupy.divide(a,b)
    def multiply(a,b):
        return cupy.multiply(a,b)
    def zeroes(shape):
        return cupy.zeros(shape)
    def average(arr,layer=0):
        return cupy.average(arr,layer)
    def matmul(arr1,arr2):
        return cupy.matmul(arr1,arr2)
    def reshape(arr,shape):
        return cupy.reshape(arr, shape)
    def default_array(inputs):
        return cupy.array(inputs)
    def get_random_method():
        return cupy.random.normal

from enum import Enum
import psutil
import multiprocessing

class ByteExpression(Enum):
    KB = 1
    MB = 2
    GB = 3
    TB = 4
    PB = 5
    
def get_cpu_count():
    return multiprocessing.cpu_count()

def get_gpu_count():
    import torch
    return torch.cuda.device_count()

def get_gpu_device_names():
    import torch
    return [torch.cuda.get_device_name(i) for i in range(get_gpu_count)]

def get_virtual_memory(byte_expr:ByteExpression = ByteExpression.GB):
    return psutil.virtual_memory().total/1024**byte_expr.value

def get_available_virtual_memory(byte_expr:ByteExpression = ByteExpression.GB):
    return psutil.virtual_memory().available/1024**byte_expr.value

def get_used_virtual_memory(byte_expr:ByteExpression = ByteExpression.GB):
    return psutil.virtual_memory().used/1024**byte_expr.value
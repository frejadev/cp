import ctypes
import numpy as np

print('p')

indata = np.ones((5,6), dtype=np.double)
outdata = np.zeros((5,6), dtype=np.double)

print(indata) 
print(outdata)

lib = ctypes.cdll.LoadLibrary('./test00.so')

print(lib.cfun)
fun = lib.cfun


fun(ctypes.c_void_p(indata.ctypes.data), 30, ctypes.c_void_p(outdata.ctypes.data))

print(indata) 
print(outdata)

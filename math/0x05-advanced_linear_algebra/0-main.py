#!/usr/bin/env python3

determinant = __import__('0-determinant').determinant

try:
    determinant(((1,),))
except TypeError as e:
    print(str(e))
try:
    determinant([(1,)])
except TypeError as e:
    print(str(e))
try:
    determinant([[1, 1], (1, 1)])
except TypeError as e:
    print(str(e))
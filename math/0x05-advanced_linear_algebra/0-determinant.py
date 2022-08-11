#!/usr/bin/env python3

def mirror(matrix, x):
  return [matrix[i][1:] for i in range(len(matrix)) if i != x]
      
def shape(matrix = []):
  if not matrix or type(matrix) != list:
    return (0, 0, 0)

  return (len(matrix), *shape(matrix[0]))

def determinant(matrix):
  if type(matrix) != list or type(matrix[0]) != list:
    raise TypeError("matrix must be a list of lists")

  first_dimention, second_dimention, *_ = shape(matrix)

  if (first_dimention == 1 and second_dimention == 1):
    return matrix[0][0]

  if (first_dimention == 1 and second_dimention == 0):
    return 1

  if first_dimention != second_dimention:
    raise ValueError("matrix must be a square matrix")

  if first_dimention == 2:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

  vector = list(zip(*matrix))[0]

  return sum(vector[i] * (-1) ** i * determinant(mirror(matrix, i)) for i in range(len(vector)))

#!/usr/bin/env python3
"""Poisson distribution"""


def factorial(number, num=None):
    """Method to calculate Factorial"""
    if num is None:
        num = number
    if num <= 1:
        return 1
    return factorial(number, num - 1) * num


class Poisson:
    """Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Constructor to poisson"""
        self.e = 2.7182818285
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            count = 0
            prob = []
            for x in data:
                count += x
            self.lambtha = float(count / len(data))

    def pmf(self, k):
        """probability mass function"""
        try:
            k = int(k)
        except Exception:
            return 0
        PMF = (self.e**(-self.lambtha) * (self.lambtha ** k)) / factorial(k)
        return PMF

    def cdf(self, k):
        """Cumulative density function"""
        try:
            k = int(k)
        except Exception:
            return 0
        CDF = 0
        for i in range(k + 1):
            CDF += (self.e**(-self.lambtha) *
                    (self.lambtha ** i)) / factorial(i)
        return CDF

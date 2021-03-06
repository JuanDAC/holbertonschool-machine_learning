#!/usr/bin/env python3
"""Binomial distribution"""


def factorial(number, num=None):
    """Method to calculate Factorial"""
    if num is None:
        num = number
    if num <= 1:
        return 1
    return factorial(number, num - 1) * num


class Binomial:
    """Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)
            self.n = round(n)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # mean = np
            # var = np(1-p)
            mean = sum(data) / len(data)
            suma = 0
            for x in data:
                suma += (x - mean) ** 2
            var = (1 / len(data)) * suma
            # np = mean
            # np(1 - p) = var
            # 1 - p = var / mean
            p = 1 - (var / mean)
            self.n = round(mean / p)
            self.p = mean / self.n

    def pmf(self, k):
        """Probability mass function"""
        try:
            if type(k) != int:
                k = int(k)
        except Exception:
            return 0
        if k < 0:
            return 0
        com = factorial(self.n) / (factorial(k) * factorial(self.n - k))
        PMF = com * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return PMF

    def cdf(self, k):
        """Probability density function"""
        try:
            if type(k) != int:
                k = int(k)
        except Exception:
            return 0
        if k < 0:
            return 0
        CDF = 0
        for i in range(k + 1):
            com = factorial(self.n) / (factorial(i) * factorial(self.n - i))
            CDF += com * (self.p ** i) * ((1 - self.p) ** (self.n - i))
        return CDF

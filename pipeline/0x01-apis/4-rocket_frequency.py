#!/usr/bin/env python3
"""
4-rocket_frequency.py
"""
from requests import get

if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches'
    response = get(url)
    data = response.json()
    rockets = {}
    for launch in data:
        rocket = launch.get('rocket')
        if rocket in rockets:
            rockets[rocket] += 1
        else:
            rockets[rocket] = 1
    for rocket, launches in sorted(rockets.items(), key=lambda x: (-x[1], x[0])):
        print('{}: {}'.format(rocket, launches))

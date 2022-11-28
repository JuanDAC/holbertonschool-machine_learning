#!/usr/bin/env python3
"""
2-user_location.py
"""
from requests import get
from sys import argv

if __name__ == '__main__':
    url = argv[1]
    response = get(url)
    data = response.json()
    if response.status_code == 403:
        print('Reset in {} min'.format(data.get('X-Ratelimit-Reset')))
    elif response.status_code == 404:
        print('Not found')
    else:
        print(data.get('location'))

#!/usr/bin/env python3
"""
3-upcoming.py
"""
from requests import get

if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    response = get(url)
    data = response.json()
    launch = data[0]
    name = launch.get('name')
    date = launch.get('date_local')
    rocket = launch.get('rocket')
    launchpad = launch.get('launchpad')
    print('{} ({}) {} - {} ({})'.format(name, date, rocket, launchpad))

#!/usr/bin/env python3
"""
0-passengers.py
"""
from requests import get


def availableShips(passengerCount):
    """
    Function that returns the list of ships that can hold a given number of
    passengers
    Arguments:
        - passengerCount: is an integer representing the number of passengers
          to search for
    Returns:
        - the list of ships that can hold a given number of passengers
    """
    url = 'https://swapi-api.hbtn.io/api/starships/'
    ships = []
    while url:
        response = get(url)
        data = response.json()
        results = data.get('results')
        for ship in results:
            passengers = ship.get('passengers')
            if passengers == 'n/a':
                continue
            elif ',' in passengers:
                passengers = int(passengers.replace(',', ''))
            else:
                passengers = int(passengers)
            if passengers >= passengerCount:
                ships.append(ship.get('name'))
        url = data.get('next')
    return ships

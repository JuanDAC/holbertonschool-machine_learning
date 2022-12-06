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
    url = "https://swapi-api.hbtn.io/api/starships/"
    r = get(url)
    json = r.json()
    results = json["results"]
    ships = []
    while json["next"]:
        for res in results:
            if res["passengers"] == 'n/a' or res["passengers"] == 'unknown':
                continue
            if int(res["passengers"].replace(',', '')) >= passengerCount:
                ships.append(res["name"])
        url = json["next"]
        r = get(url)
        json = r.json()
        results = json["results"]
    return ships

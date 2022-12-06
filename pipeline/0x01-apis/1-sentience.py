#!/usr/bin/env python3
"""
1-sentience.py
"""
from requests import get


def sentientPlanets():
    """
    Function that returns the list of names of the home planets of all sentient species.
    species
    Returns:
        - the list of names of the home planets of all sentient species
    """
    planets = []
    url = 'https://swapi-api.hbtn.io/api/species/'
    response = get(url)  # No authentication is required
    r_code = response.status_code
    next_url = url
    while (r_code == 200):

        json_data = response.json()['results']

        for specie in json_data:
            if (specie['designation'] == "sentient" or
               specie["classification"].lower() == "sentient"):
                url_planet = specie['homeworld']
                if (url_planet is not None):
                    planets.append(get(url_planet).json()['name'])

        next_url = response.json()['next']
        if (next_url is None):
            break

        response = get(next_url)

    return (planets)

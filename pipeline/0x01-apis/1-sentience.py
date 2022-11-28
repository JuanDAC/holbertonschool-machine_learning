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
    url = 'https://swapi-api.hbtn.io/api/species/'
    planets = []
    while url:
        response = get(url)
        data = response.json()
        results = data.get('results')
        for specie in results:
            classification = specie.get('classification')
            if classification == 'sentient':
                planets.append(specie.get('homeworld'))
        url = data.get('next')
    return planets

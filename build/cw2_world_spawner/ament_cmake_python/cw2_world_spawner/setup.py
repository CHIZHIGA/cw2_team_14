from setuptools import find_packages
from setuptools import setup

setup(
    name='cw2_world_spawner',
    version='0.1.0',
    packages=find_packages(
        include=('cw2_world_spawner', 'cw2_world_spawner.*')),
)

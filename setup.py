import os.path as osp
from distutils.core import setup
from setuptools import find_packages

def get_requirements(filename='requirements.txt'):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires

setup(
  name = 'rag_analyzer',         
  packages = find_packages(),
  version = '0.1',      
  license='apache-2.0',        
  description = 'Python module providing tools for analyzing documents.',   
  author = 'Frederik Polachowski',
  install_requires=get_requirements()
)
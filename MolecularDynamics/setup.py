from setuptools import setup, find_packages

setup(
    name='trapped_atoms_simulation',  
    version='0.1.0',
    author='Paul Christ', 
    description='Simulationscode für meine Masterarbeit',

    packages=find_packages(),
    
   
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'pykeops' 

    ],
)

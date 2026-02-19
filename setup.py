from setuptools import find_packages, setup
from typing import List

# Constant to handle the editable install flag in requirements.txt
HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns a list of requirements from a file.
    """
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        # Remove the newline characters
        requirements = [req.replace('\n', '') for req in requirements]

        # Remove '-e .' if it exists so it doesn't try to install it as a package
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name='ml_project',
    version='0.0.1',
    author='random65',
    author_email='hardstuckbronze10.com', # Good practice to include this
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)


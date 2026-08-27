from setuptools import find_packages, setup
from typing import List

def get_requirements()->List[str]:
    """
    this function returms the list of requirements from requirements.txt 
    """

    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt', 'r') as file:
            ##read lines from the file
            lines=file.readlines()
            ##process lines
            for line in lines:
                #remove spaces
                requirement=line.strip()
                ## ignore empty lines
                ##and ignore -e.
                if requirement and requirement!='-e .':
                    requirement_lst.append(requirement)
                    ##make a list of all the packages you need to install
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

setup(
    name="compiler_optimization",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requirements()
)
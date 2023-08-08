from setuptools import setup
from setuptools import setup, find_packages

setup(
   name='cloth_training',
   version='1.0',
   description='A useful module',
   author='Man Foo',
   author_email='foomail@foo.example',
   #packages=find_packages(),
   packages = ['cloth_training'],
   install_requires=[
                       
                     ], #external packages as dependencies
)




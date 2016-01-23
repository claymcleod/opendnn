from setuptools import setup, find_packages

setup(name='opendnn',
      version='1.0.0',
      description='Open deep neural network framework',
      author='Clay McLeod',
      author_email='clay.l.mcleod@gmail.com',
      url='https://github.com/claymcleod/opendnn',
      license='MIT',
      install_requires=['theano'],
      packages=find_packages()
      )

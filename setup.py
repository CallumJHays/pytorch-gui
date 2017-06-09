
from setuptools import setup, find_packages

setup(name='PytorchGui',
      version='0.1',
      description='GUI interface to PyTorch',
      author='Callum Hays',
      author_email='callum@wearepopgun.com',
      url='https://popgun.ai/',
      packages=find_packages(),
      package_data={
          'pytorchgui': ['*.html']
      },
      include_package_data=True
      )

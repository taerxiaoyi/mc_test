from setuptools import setup, find_packages

setup(name='westlake_sdkpy',
      version='1.0.1',
      author='WLRobotics',
      author_email='19179921356@163.com',
      license="BSD-3-Clause",
      packages=find_packages(),      
      description='WestLake robot sdk version 1 for python',
      python_requires='>=3.8',
      install_requires=[
            "cyclonedds==0.10.2",
            "numpy",
            "opencv-python",
            'pyyaml',
      ],
      )
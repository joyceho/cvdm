from setuptools import setup

setup(name='cvdm',
      version='0.1',
      description='Scripts for assessing cardiovscular risk in patients with diabetes',
      url='http://github.com/joyceho/cvdm',
      author='Joyce Ho',
      author_email='joyce.c.ho@emory.edu',
      license='MIT',
      packages=['cvdm'],
      install_requires=[
                       'numpy',
                       'pytest',
                     ],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      zip_safe=False)
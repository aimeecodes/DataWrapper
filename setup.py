from setuptools import setup

setup(
    name='DataWrapper',
    version='0.1.0',
    author='Aimee Johnston',
    author_email='source@aimee.codes',
    packages=['DataWrapper'],
    # scripts=['bin/script1','bin/script2'],
    # url='http://pypi.python.org/pypi/PackageName/',
    # license='LICENSE.txt',
    description='Wrapper for operations using popular machine learning packages',
    long_description=open('README.org').read(),
    install_requires=['pandas',
                      'sklearn',
                      'numpy',
                      'matplotlib',
                      'seaborn',
                      'scipy',
                      'wheel']
)

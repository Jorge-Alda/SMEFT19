from setuptools import setup

with open('README.md', 'r') as f:
	long_description = f.read()

setup(
	name='SMEFT19',
	version='3.0.1',
	description='Global likelihood study with the SMEFT operators Olq1 and Qlq3',
	license='MIT',
	author='Jorge Alda',
	author_email='jalda@unizar.es',
	url='https://github.com/Jorge-Alda/SMEFT19',
	packages=['SMEFT19'],
)

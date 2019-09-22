from setuptools import setup

setup(
    name='gym_taxi',
    version='0.0.4',
    description='Gym taxi environment - useful to replicate Random Walk experiments',
    author='Miguel Morales',
    author_email='mimoralea@gmail.com',
    packages=['gym_taxi', 'gym_taxi.envs'],
    license='MIT License',
    install_requires=['gym'],
)

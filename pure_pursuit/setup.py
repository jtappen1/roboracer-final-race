from glob import glob
import os

from setuptools import setup

package_name = 'final_race_pure_pursuit'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'path'), glob('path/*.csv'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zzangupenn, Hongrui Zheng',
    maintainer_email='zzang@seas.upenn.edu, billyzheng.bz@gmail.com',
    description='roboracer final race pure pursuit',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'final_race_pure_pursuit_node = final_race_pure_pursuit.pure_pursuit_node:main',
        ],
    },
)

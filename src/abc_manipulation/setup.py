# abc_manipulation/setup.py

import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'abc_manipulation'

config_files = glob('config/*')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
     
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), ['abc_manipulation/T_gripper2camera.npy']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), ['config/ .env']),
    ],
    
    package_data={
        package_name: ['*.npy'],
    },
    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ssu',
    maintainer_email='ssu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'test_server = abc_manipulation.test_server:main',
            'task_planner = abc_manipulation.task_planner:main',
            'place_item = abc_manipulation.place_item:main',
            'pick_item = abc_manipulation.pick_item:main',
            'scan_barcode = abc_manipulation.scan_barcode:main',
        ],
    },
)
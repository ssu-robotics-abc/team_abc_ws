# abc_manipulation/setup.py

from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'abc_manipulation'

setup(
    name=package_name,
    version='0.0.0',
    # 패키지 폴더 내의 __init__.py가 있는 곳들을 찾습니다.
    packages=find_packages(exclude=['test']),
    
    # [수정된 부분] 
    # 1. data_files는 ROS 시스템에서 필요한 설정파일을 옮길 때 씁니다.
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    
    # 2. package_data를 사용하여 .npy 파일을 파이썬 패키지 폴더 안에 포함시킵니다.
    # 이렇게 해야 site-packages/abc_manipulation/ 폴더 안에 npy가 들어갑니다.
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
        ],
    },
)
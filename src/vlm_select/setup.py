from setuptools import find_packages, setup
from glob import glob
import os # os 모듈 추가 필요

package_name = 'vlm_select'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        (
            'share/' + package_name + '/config',
            glob('config/*.yaml') + glob('vlm_select/Calibration_Tutorial/*.npy')
        ),
        ('share/' + package_name, glob('best.pt')),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'requests', 'opencv-python', 'ultralytics', 'google-generativeai'],
    zip_safe=True,
    maintainer='deeptree',
    maintainer_email='deeptree@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'stt_node = dsr_practice.stt_node:main',
            'stt_robot_control = dsr_practice.stt_robot_control:main',
            'stt_pick_and_place = dsr_practice.stt_pick_and_place:main',
            'vlm_test = dsr_practice.snack_pick:main',
            'vlm_api_node = dsr_practice.vlm_api_node:main',
            'vlm_command_node = dsr_practice.vlm_command_node:main',
            'yolo_detector_node = dsr_practice.yolo_detector_node:main',
            'vlm_api_yolo = dsr_practice.vlm_api_yolo:main',
        ],
    },
)

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
            'stt_node = vlm_select.stt_node:main',
            'stt_robot_control = vlm_select.stt_robot_control:main',
            'stt_pick_and_place = vlm_select.stt_pick_and_place:main',
            'vlm_test = vlm_select.snack_pick:main',
            'vlm_api_node = vlm_select.vlm_api_node:main',
            'vlm_command_node = vlm_select.vlm_command_node:main',
            'yolo_detector_node = vlm_select.yolo_detector_node:main',
            'vlm_api_yolo = vlm_select.vlm_api_yolo:main',
        ],
    },
)

import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'lidarscan_msg_yolo_demo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python-headless>=4.5',
        'ultralytics>=8.3.0',
        'torch',
    ],
    zip_safe=True,
    maintainer='ouster developers',
    maintainer_email='oss@ouster.io',
    description='YOLO inference on LidarScan RANGE / NEAR_IR / REFLECTIVITY channels',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidarscan_yolo_consumer = lidarscan_msg_yolo_demo.lidarscan_yolo_consumer:main',
        ],
    },
)

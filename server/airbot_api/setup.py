import os
from setuptools import setup, find_packages

# 获取 setup.py 所在目录
here = os.path.abspath(os.path.dirname(__file__))

setup(
    name='airbot_sdk',
    version='0.1.0',
    setup_requires=['setuptools>=40.0'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pyrealsense2',
        'airbot_py'
    ],
    entry_points={},
    python_requires='>=3.6',
    zip_safe=False
)

from setuptools import setup
import os
from glob import glob

package_name = 'aida_core'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'PyWavelets', 'numpy', 'opencv-python'],
    zip_safe=True,
    maintainer='AIDA Developer',
    maintainer_email='aida@example.com',
    description='AIDA core package containing the impact verification node and other enhancements.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'impact_verifier_node = aida_core.impact_verifier_node:main',
            'aida_vidar_node = aida_core.aida_vidar_node:main',
            'aida_memory_node = aida_core.aida_memory_node:main'
        ],
    },
)

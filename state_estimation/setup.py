from glob import glob

from setuptools import setup

package_name = 'state_estimation'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/estimators', glob('config/estimators/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jacob Bauman',
    maintainer_email='noreply@example.com',
    description='Switchable state estimation for the RoboRacer MPC',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_prep = state_estimation.sensor_prep_node:main',
            'state_assembler = state_estimation.state_assembler_node:main',
            'shadow_ekf1 = state_estimation.shadow_ekf1_node:main',
            'viz_tf = state_estimation.viz_tf_node:main',
        ],
    },
)

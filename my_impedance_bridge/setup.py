from setuptools import setup

package_name = 'my_impedance_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='you@example.com',
    description='Bridge node: streams MoveIt-planned trajectories into franka_example_controllers/JointImpedanceController',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # format: '<executable_name> = <package>.<module>:<function>'
            # this is what makes `ros2 run my_impedance_bridge impedance_bridge_node` work
            'impedance_bridge_node = my_impedance_bridge.impedance_bridge_node:main',
        ],
    },
)

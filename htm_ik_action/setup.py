from setuptools import find_packages, setup

package_name = 'htm_ik_action'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='you@example.com',
    description='Action server: HTM in, straight-line Cartesian goal via MoveIt out',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'htm_ik_server = htm_ik_action.htm_ik_server:main',
            'htm_ik_server_chomp = htm_ik_action.htm_ik_server_chomp:main',
            'htm_ik_server_chomp_seeded = htm_ik_action.htm_ik_server_chomp_seeded:main',
        ],
    },
)

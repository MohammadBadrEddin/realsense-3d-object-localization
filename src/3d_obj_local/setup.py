from setuptools import find_packages, setup

package_name = '3d_obj_local'

setup(
    name=package_name,
    version='1.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['3d_obj_local/best.pt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='darkpassenger',
    maintainer_email='darkpassenger@todo.todo',
    description='Vision node using YOLO',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'vision_node = 3d_obj_local.vision_node:main',
        ],
    },
)

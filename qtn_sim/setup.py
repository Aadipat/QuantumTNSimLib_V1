from setuptools import setup, find_packages

setup(
    name='qtn_sim',
    version='0.1.0',
    description='My qtn_sim package v1!',
    author='Aadi Patwardhan',
    package_dir={'': 'src'},
    packages=find_packages(where='src', include=['qtn_sim', 'qtn_sim.*']),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'opt_einsum',
        'pytest'
    ]
    ,
    include_package_data=True,
)

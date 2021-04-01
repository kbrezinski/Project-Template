from setuptools import setup

setup(
    name='projectTemplate',
    version='0.1',
    description='project template for larger code base projects',
    keywords='practice template',
    classifiers=[
        'Programing Language :: Python',
        'Programing Language :: Python :: 3'
    ],
    author='Kenneth Brezinski',
    author_email='brezinkk@myumanitoba.ca',
    packages=['projectTemplate'],
    install_requires=['numpy>=1.11.0', 'scipy>=0.17.0'],
    test_suite='nose.collector',
    tests_require=['nose>=1.3.1'],
    include_package_data=True,
    zip_safe=False
)
from distutils.core import setup

setup(
    name='trgpy',
    version='0.1dev',
    author='Thomas R. Greve',
    author_email='t.r.greve@gmail.com',
    package_dir={'trgpy': 'src'},
    packages=['trgpy'],
    long_description=open('README.txt').read(),
)

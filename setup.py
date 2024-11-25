from setuptools import setup
from pathlib import Path

# Read the description
readme_file = Path(__file__).absolute().parent / "README.md"
with readme_file.open("r") as fp:
    long_description = fp.read()
    
setup(
    name='hydroecolstm',
    version='0.3.0',    
    description='A python package for HydroEcological Modelling using LSTM',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tamnva/hydroecolstm',
    author='Tam V. Nguyen',
    author_email='tamnva@gmail.com',
    packages=['hydroecolstm', 
              'hydroecolstm.data',
              'hydroecolstm.interface', 
              'hydroecolstm.utility', 
              'hydroecolstm.model',
              'hydroecolstm.train'],
    python_requires='>=3.8',
    install_requires=['pandas',
                      'numpy',
                      'torch',
                      'pandastable',
                      'tkcalendar',
                      'PyYAML',
                      'pathlib',
                      'customtkinter',
                      'CTkToolTip',
                      'CTkMessagebox',
                      'CTkListbox',
                      'matplotlib',
                      'ray[tune]'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
    ],

    include_package_data=True,
    package_data={'hydroecolstm': ['images/*']},
)

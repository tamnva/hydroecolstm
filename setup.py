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
    install_requires=['pandas>=2.2.0',
                      'numpy>=1.24.4',
                      'pytorch>=2.3.0',
                      'pandastable>=0.13.1',
                      'tkcalendar>=1.6.1',
                      'PyYAML>=6.0.1',
                      'pathlib>=1.0.1',
                      'customtkinter>=5.2.1',
                      'CTkToolTip>=0.8',
                      'CTkMessagebox>=2.5',
                      'CTkListbox>=0.10',
                      'matplotlib>=3.9.2',
                      'ray>=2.39.0'
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

from setuptools import setup

setup(
    name='hydroecolstm',
    version='0.1.0',    
    description='A python package for HydroEcological Modelling using LSTM',
    url='https://github.com/tamnva/HydroPyTorch',
    author='Tam V. Nguyen',
    author_email='tamnva@gmail.com',
    license=' GPL-3.0',
    packages=['hydroecolstm', 
              'hydroecolstm.data',
              'hydroecolstm.interface', 
              'hydroecolstm.utility', 
              'hydroecolstm.model'],
    python_requires='>=3.8',
    install_requires=['pandas',
                      'numpy',
                      'torch',
                      'customtkinter',
                      'CTkToolTip',
                      'datetime',
                      'pandastable',
                      'tkcalendar',
                      'CTkListbox',
                      'CTkToolTip',
                      'ruamel.yaml'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GPL-3.0 license ',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
    ],
)

# Configuration file for the Sphinx documentation builder.

# Import required library
import hydroecolstm
import datetime
import os
import sys

#source code directory, relative to this file, for sphinx-autobuild
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information
project = 'HydroEcoLSTM'
copyright = f'{datetime.datetime.now().year}, {hydroecolstm.__author__}'
author = hydroecolstm.__author__

release = '0.1'
version = hydroecolstm.__version__

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'


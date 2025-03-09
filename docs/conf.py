# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import stanford_theme
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

master_doc = 'index'

# -- Project information -----------------------------------------------------

project = 'desolver'
copyright = '2021, Ekin Ozturk'
author = 'Ekin Ozturk'

# The full version, including alpha/beta/rc tags
release = '5.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "sphinx.ext.napoleon",
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'IPython.sphinxext.ipython_console_highlighting',
#     "sphinx.ext.intersphinx" 
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'setup.py', '**/*backend*', '**/.ipynb_checkpoints/*']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_theme
    html_theme = 'stanford_theme'
    html_theme_path = [sphinx_theme.get_html_theme_path('stanford-theme')]
else:
    html_theme_options = {
        "page_width": "75%",
        "sidebar_width": "5%"
    }

# otherwise, readthedocs.org uses their theme by default, so no need to specify it

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# intersphinx_mapping = {'audi': ("https://darioizzo.github.io/audi/", None)}

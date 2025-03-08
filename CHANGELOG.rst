Changelog
=========

Version 5.0.0
-------------
* Fully ported backend to use autoray instead of manual specification of each backend, this enables a more portable setup
* Fully removed pyaudi as a backend

Version 4.5.0
-------------
* Updated to `pyproject.toml` setup and removed `pyaudi` support.
* Added changelog to repository
* Moved to pyproject.toml and hatch build system
* Updated github workflow to use hatch
* Reconfigured tests to have skip flags and have tests enabled by default

Version 4.2.0
-------------
* Improved performance of implicit methods, added embedded implicit methods following Kroulíková (2017) for fully implicit adaptive integration.

Version 4.1.0
-------------
* Initial release of implicit integration schemes that use a basic newton-raphson algorithm to solve for the intermediate states.

Version 3.0.0
-------------
* PyAudi support has been finalised. It is now possible to do numerical integrations using ``gdual`` variables such as ``gdual_double``\ , ``gdual_vdouble`` and ``gdual_real128`` (only on select platforms, refer to `pyaudi docs <https://darioizzo.github.io/audi/>`_ for more information). Install desolver with pyaudi support using ``pip install desolver[pyaudi]``. Documentation has also been added and is available at `desolver docs <https://desolver.readthedocs.io/>`_.

Version 2.5.0
-------------
* Event detection has been added to the module. It is now possible to do numerical integration with terminal and non-terminal events.

Version 2.2.0
-------------
* PyTorch backend is now implemented. It is now possible to numerically integrate a system of equations that use pytorch tensors and then compute gradients from these.


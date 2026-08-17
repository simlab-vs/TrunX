Installation
============

.. code-block:: bash

   uv sync

The R comparison interface (``trunx.gp3.run_r3pg``) is not included in
this release. If you need it, install the required system packages
(``r-base-dev``, ``libtirpc-dev``) and run:

.. code-block:: bash

   uv sync --extra r-interface

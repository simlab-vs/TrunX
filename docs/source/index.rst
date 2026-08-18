trunx documentation
====================

Minimal release of the 3PG forestry model, containing only what is
needed to run it via ``scripts/run_3pg_main.py``.

Features
--------

- Pure JAX implementation of the 3PG forest growth model — jit- and
  autodiff-compatible
- Multi-species, monthly time-step simulation of stand biomass, LAI,
  DBH, and other 3PG state variables
- Automatic differentiation of simulation outputs with respect to
  model parameters, via :func:`jax.jacobian`
- Reads site, climate, species, and parameter inputs directly from an
  Excel workbook
- Built-in plotting of simulation outputs

Quick Example
-------------

The minimal entrypoint this release is built around:

.. literalinclude:: ../../scripts/run_3pg_main.py
   :language: python

Installation
------------

.. code-block:: bash

   uv sync

The R comparison interface (``trunx.gp3.run_r3pg``) is not included in
this release. If you need it, install the required system packages
(``r-base-dev``, ``libtirpc-dev``) and run:

.. code-block:: bash

   uv sync --extra r-interface

Usage
-----

Place your input workbook at ``data/threepg_inputs/solling_data.xlsx``
(not included — supply your own; see
:func:`trunx.gp3.PG3_model_impl.prepare_data` for the expected
``site``/``climate``/``species``/``parameters``/``observed`` sheet
structure), then run:

.. code-block:: bash

   uv run python scripts/run_3pg_main.py

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   User Guide <user_guide>
   Advanced Usage <advanced_usage>
   API Reference <api/modules>

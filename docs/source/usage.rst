Usage
=====

Place your input workbook at ``data/threepg_inputs/solling_data.xlsx``
(not included — supply your own; see
:func:`trunx.gp3.PG3_model_impl.prepare_data` for the expected
``site``/``climate``/``species``/``parameters``/``observed`` sheet
structure), then run:

.. code-block:: bash

   uv run python scripts/run_3pg_main.py

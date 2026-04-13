Installation instructions
=========================

Prerequisites
-------------

``pylcp`` requires Python >= 3.11.

As of v2.0, ``pylcp`` uses `JAX <https://github.com/jax-ml/jax>`_ and
`Diffrax <https://github.com/patrick-kidger/diffrax>`_ as its numerical backend.

Stable release (v1.x)
----------------------

The stable release is available on PyPI::

  pip install pylcp

.. note::

   The PyPI version (1.x) still uses the original scipy backend. v2.0 has not
   yet been released on PyPI.

Development version (v2.0, JAX backend)
---------------------------------------

Using `uv <https://docs.astral.sh/uv/>`_ (recommended)::

  git clone https://github.com/palmispojk/pylcp/
  cd pylcp
  uv sync

Or using pip::

  git clone https://github.com/palmispojk/pylcp/
  cd pylcp
  pip install .

GPU support
-----------

For GPU-accelerated simulations (requires a CUDA-capable GPU):

Using uv::

  uv sync --extra cuda

Using pip::

  pip install ".[cuda]"

This installs JAX with CUDA 12 support. GPU tests in the test suite are
automatically skipped when no GPU is detected.

Development setup
-----------------

To install development and documentation dependencies::

  uv sync --group dev --group docs

Run the test suite with::

  uv run pytest

See `CONTRIBUTING.md <https://github.com/JQIamo/pylcp/blob/master/CONTRIBUTING.md>`_
for code style and linting instructions.
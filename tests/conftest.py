"""Pytest conftest — runs before any test module is imported.

Sets XLA_FLAGS to create 4 virtual CPU devices so that multi-device
sharding tests (test_multi_device_sharding.py) work even when collected
alongside other test files.  This has no effect on single-device tests;
they continue to use jax.devices('cpu')[0] as before.

The flag is only added when no --xla_force_host_platform_device_count is
already present in XLA_FLAGS, so user-specified values are respected.
"""
import os

# Must happen before JAX is imported anywhere.
_existing = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _existing:
    os.environ["XLA_FLAGS"] = f"{_existing} --xla_force_host_platform_device_count=4".strip()

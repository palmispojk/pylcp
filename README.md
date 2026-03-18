pylcp
=================================

[![Documentation Status](https://readthedocs.org/projects/python-laser-cooling-physics/badge/?version=latest)](https://python-laser-cooling-physics.readthedocs.io/en/latest/?badge=latest) [![GitHub version](https://badge.fury.io/gh/jqiAMO%2Fpylcp.svg)](https://badge.fury.io/gh/jqiAMO%2Fpylcp) [![PyPI version](https://badge.fury.io/py/pylcp.svg)](https://badge.fury.io/py/pylcp) [![Google group : SSFAM News](https://img.shields.io/badge/Google%20Group-pylcp-blue.svg)](https://groups.google.com/g/pylcp)

`pylcp` is a Python package for calculating laser cooling physics.
It automatically generates optical Bloch equations (or approximations thereof)
given an atom's or molecule's internal Hamiltonian, a set of laser beams, and
a magnetic field.

As of v2.0, `pylcp` uses [JAX](https://github.com/jax-ml/jax) and
[Diffrax](https://github.com/patrick-kidger/diffrax) as its numerical backend,
providing JIT compilation and GPU acceleration for significantly faster
simulations. Batched trajectory evolution allows multiple atoms to be simulated
in parallel.

> **Note:** v2.0 is currently in development and has not yet been released on
> PyPI. The PyPI version (1.x) still uses the original scipy backend.

If you find `pylcp` useful in your research, please cite our paper describing the package: [![DOI](http://img.shields.io/badge/Computer%20Physics%20Communications-j.cpc.2021.108166-lightblue.svg)](https://doi.org/10.1016/j.cpc.2021.108166).

Installation
------------
Requires Python >= 3.11.

The stable release (v1.x, scipy backend) is available on PyPI:
```
pip install pylcp
```

To install the development version (v2.0, JAX backend), clone the repository:
```
git clone https://github.com/JQIamo/pylcp.git
cd pylcp
pip install -e .
```

To also install development and documentation dependencies:
```
pip install -e . --group dev --group docs
```

Basic Usage
-----------
The basic workflow for `pylcp` is to define the elements of the problem (the
laser beams, magnetic field, and Hamiltonian), combine these together in a
governing equation, and then calculate something of interest.

The first step is to define the problem.
The Hamiltonian is represented as a series of blocks, which we first define
individually. For this example, we assume a ground state (`g`) and an excited
state (`e`) with some detuning `delta`:
```python
Hg = np.array([[0.]])
He = np.array([[-delta]])
mu_q = np.zeros((3, 1, 1))
d_q = np.zeros((3, 1, 1))
d_q[1, 0, 0] = 1.

hamiltonian = pylcp.hamiltonian(Hg, He, mu_q, mu_q, d_q, mass=mass)
```

Next, define the laser beams. Here we create two counterpropagating beams:
```python
laserBeams = pylcp.laserBeams([
    {'kvec': np.array([1., 0., 0.]), 'pol': np.array([0., 1., 0.]),
     'pol_coord': 'spherical', 'delta': delta, 's': norm_intensity},
    {'kvec': np.array([-1., 0., 0.]), 'pol': np.array([0., 1., 0.]),
     'pol_coord': 'spherical', 'delta': delta, 's': norm_intensity}
], beam_type=pylcp.infinitePlaneWaveBeam)
```

And the magnetic field (here a quadrupole field):
```python
magField = pylcp.quadrupoleMagneticField(alpha)
```

Combine the components into a governing equation:
```python
obe = pylcp.obe(laserBeams, magField, hamiltonian)
```

Then calculate quantities of interest. For example, to compute the force at
positions `R` and velocities `V`:
```python
obe.generate_force_profile(R, V)
```

Or evolve the motion of one or more atoms:
```python
obe.evolve_motion([0, t_final], freeze_axis=[True, True, False])
```

There are plenty of examples in the `docs/examples/` directory as Jupyter
notebooks.

Further documentation is available at https://python-laser-cooling-physics.readthedocs.io.

In Development
--------------
The v2.0 branch is under active development. Key changes from v1.x:

- **JAX/Diffrax backend** replacing scipy for all ODE integration
- **JIT-compiled** force, density matrix, and equation-of-motion evaluations
- **Batched `evolve_motion`** for simulating multiple atoms in a single call
- **GPU-accelerated `evolve_motion`** using `jax.vmap` for batched atom trajectories
- **Improved convergence** with a monotonic decay guard for dark-state detection in `generate_force_profile`

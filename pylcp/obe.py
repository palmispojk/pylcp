"""
Tools for solving the OBE for laser cooling
author: spe
"""
import functools
import numpy as np
import jax
import jax.numpy as jnp
from .integration_tools_gpu import solve_ivp_random, solve_ivp_dense

from .rateeq import rateeq
from .common import (cart2spherical, spherical2cart, base_force_profile)
from .governingeq import governingeq



class force_profile(base_force_profile):
    """
    Optical Bloch equation force profile

    The force profile object stores all of the calculated quantities created by
    the rateeq.generate_force_profile() method.  It has the following
    attributes:

    Attributes
    ----------
    R  : array_like, shape (3, ...)
        Positions at which the force profile was calculated.
    V : array_like, shape (3, ...)
        Velocities at which the force profile was calculated.
    F : array_like, shape (3, ...)
        Total equilibrium force at position R and velocity V.
    f_mag : array_like, shape (3, ...)
        Magnetic force at position R and velocity V.
    f : dictionary of array_like
        The forces due to each laser, indexed by the
        manifold the laser addresses.  The dictionary is keyed by the transition
        driven, and individual lasers are in the same order as in the
        pylcp.laserBeams object used to create the governing equation.
    f_q : dictionary of array_like
        The force due to each laser and its :math:`q` component, indexed by the
        manifold the laser addresses.  The dictionary is keyed by the transition
        driven, and individual lasers are in the same order as in the
        pylcp.laserBeams object used to create the governing equation.
    Neq : array_like
        Equilibrium population found.
    """
    def __init__(self, R, V, laserBeams, hamiltonian):
        super().__init__(R, V, laserBeams, hamiltonian)

        self.iterations = np.zeros(self.R[0].shape, dtype='int64')
        self.fq = {}
        for key in laserBeams:
            self.fq[key] = np.zeros(self.R.shape + (3, len(laserBeams[key].beam_vector)))

    def store_data(self, ind, Neq, F, F_laser, F_mag, iterations, F_laser_q):
        super().store_data(ind, Neq, F, F_laser, F_mag)

        for jj in range(3):
            for key in F_laser_q:
                self.fq[key][(jj,) + ind] = F_laser_q[key][jj]

        self.iterations[ind] = iterations


class obe(governingeq):
    """
    The optical Bloch equations

    This class constructs the optical Bloch equations from the given laser
    beams, magnetic field, and hamiltonian.

    Parameters
    ----------
    laserBeams : dictionary of pylcp.laserBeams, pylcp.laserBeams, or list of pylcp.laserBeam
        The laserBeams that will be used in constructing the optical Bloch
        equations.  which transitions in the block diagonal hamiltonian.  It can
        be any of the following:

            * A dictionary of pylcp.laserBeams: if this is the case, the keys of
              the dictionary should match available :math:`d^{nm}` matrices
              in the pylcp.hamiltonian object.  The key structure should be
              `n->m`.
            * pylcp.laserBeams: a single set of laser beams is assumed to
              address the transition `g->e`.
            * a list of pylcp.laserBeam: automatically promoted to a
              pylcp.laserBeams object assumed to address the transtion `g->e`.

    magField : pylcp.magField or callable
        The function or object that defines the magnetic field.
    hamiltonian : pylcp.hamiltonian
        The internal hamiltonian of the particle.
    a : array_like, shape (3,), optional
        A default acceleraiton to apply to the particle's motion, usually
        gravity. Default: [0., 0., 0.]
    transform_into_re_im : boolean
        Optional flag to transform the optical Bloch equations into real and
        imaginary components.  This helps to decrease computaiton time as it
        uses the symmetry :math:`\\rho_{ji}=\\rho_{ij}^*` to cut the number
        of equations nearly in half.  Default: True
    use_sparse_matrices : boolean or None
        Optional flag to use sparse matrices.  If none, it will use sparse
        matrices only if the number of internal states > 10, which would result
        in the evolution matrix for the density operators being a 100x100
        matrix.  At that size, there may be some speed up with sparse matrices.
        Default: None
    include_mag_forces : boolean
        Optional flag to inculde magnetic forces in the force calculation.
        Default: True
    r0 : array_like, shape (3,), optional
        Initial position.  Default: [0., 0., 0.]
    v0 : array_like, shape (3,), optional
        Initial velocity.  Default: [0., 0., 0.]

    Methods
    -------
    """
    def __init__(self, laserBeams, magField, hamitlonian,
                 a=jnp.array([0., 0., 0.]), transform_into_re_im=True, include_mag_forces=True,
                 r0=jnp.array([0., 0., 0.]), v0=jnp.array([0., 0., 0.])):

        super().__init__(laserBeams, magField, hamitlonian, a=a,
                         r0=r0, v0=v0)

        # Save the optional arguments:
        self.transform_into_re_im = transform_into_re_im
        self.include_mag_forces = include_mag_forces

        # Set up a dictionary to store any resulting force profiles.
        self.profile = {}

        # Reset the current solution to None
        self.sol = None

        # There will be time-dependent and time-independent components of the optical
        # Bloch equations.  The time-independent parts are related to spontaneous
        # emission, applied magnetic field, and the zero-field Hamiltonian.  We
        # compute the latter-two directly from the commuatator.

        # Build the matricies that control evolution:
        self.ev_mat = {}
        self.__build_decay_ev()
        self.__build_coherent_ev()

        # If necessary, transform the evolution matrices:
        if self.transform_into_re_im:
            self.__transform_ev_matrices()
            
        self.__cast_ev_mat_to_jax()
    
    
    @property
    def magField(self):
        return self._magField

    @magField.setter
    def magField(self, value):
        self._magField = value
        # Changing the field invalidates the cached _dydt closure so JAX
        # retraces and compiles a new XLA kernel with the updated field.
        self.__dict__.pop('_dydt', None)

    @property
    def laserBeams(self):
        return self._laserBeams

    @laserBeams.setter
    def laserBeams(self, value):
        self._laserBeams = value
        self.__dict__.pop('_dydt', None)

    def __cast_ev_mat_to_jax(self):
        """Recursively convert the nested dictionaries of numpy arrays to jax arrays"""
        for key in self.ev_mat:
            if isinstance(self.ev_mat[key], dict):
                for subkey in self.ev_mat[key]:
                    self.ev_mat[key][subkey] = jnp.asarray(self.ev_mat[key][subkey], dtype=jnp.complex128)
            elif isinstance(self.ev_mat[key], list):
                self.ev_mat[key] = [jnp.asarray(v, dtype=jnp.complex128) for v in self.ev_mat[key]]
            else:
                self.ev_mat[key] = jnp.asarray(self.ev_mat[key], dtype=jnp.complex128)




    def __density_index(self, ii, jj):
        """
        This function returns the index in the rho vector that corresponds to element rho_{ij}.  If
        """
        return ii + jj * self.hamiltonian.n


    def __build_coherent_ev_submatrix(self, H):
        """
        This method builds the coherent evolution based on a submatrix of the
        Hamiltonian H.  In practice, one must be careful about commutators if
        one breaks up the Hamiltonian.
        """
        ev_mat = np.zeros((self.hamiltonian.n**2, self.hamiltonian.n**2),
                               dtype='complex128')

        for ii in range(self.hamiltonian.n):
            for jj in range(self.hamiltonian.n):
                for kk in range(self.hamiltonian.n):
                    ev_mat[self.__density_index(ii, jj),
                           self.__density_index(ii, kk)] += 1j*H[kk, jj]
                    ev_mat[self.__density_index(ii, jj),
                           self.__density_index(kk, jj)] -= 1j*H[ii, kk]

        return ev_mat

    # Is only used in construction so can remain in np
    def __build_coherent_ev(self):
        self.ev_mat['H0'] = self.__build_coherent_ev_submatrix(
            np.array(self.hamiltonian.H_0)
        )

        self.ev_mat['B'] = [None]*3
        for q in range(3):
            self.ev_mat['B'][q] = self.__build_coherent_ev_submatrix(
                np.array(self.hamiltonian.mu_q[q])
            )

        self.ev_mat['d_q'] = {}
        self.ev_mat['d_q*'] = {}
        for key in self.laserBeams.keys():
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['gamma']
            self.ev_mat['d_q'][key] = [None]*3
            self.ev_mat['d_q*'][key] = [None]*3
            for q in range(3):
                self.ev_mat['d_q'][key][q] = self.__build_coherent_ev_submatrix(
                    gamma*self.hamiltonian.d_q_bare[key][q]/4.
                )
                self.ev_mat['d_q*'][key][q] = self.__build_coherent_ev_submatrix(
                    gamma*self.hamiltonian.d_q_star[key][q]/4.
                )
            self.ev_mat['d_q'][key] = np.array(self.ev_mat['d_q'][key])
            self.ev_mat['d_q*'][key] = np.array(self.ev_mat['d_q*'][key])

    # is used only in construction
    def __build_decay_ev(self):
        """
        This method constructs the decay portion of the OBE using the radiation
        reaction approximation.
        """
        d_q_bare = self.hamiltonian.d_q_bare
        d_q_star = self.hamiltonian.d_q_star

        self.decay_rates = {}
        self.decay_rates_truncated = {}
        self.decay_rho_indices = {}
        self.recoil_velocity = {}

        self.ev_mat['decay'] = np.zeros((self.hamiltonian.n**2,
                                         self.hamiltonian.n**2),
                                        dtype='complex128')

        # Go through each dipole moment and calculate:
        for key in d_q_bare:
            ev_mat = np.zeros((self.hamiltonian.n**2, self.hamiltonian.n**2),
                               dtype='complex128')
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['gamma']

            # The first index we want to capture:
            for ii in range(self.hamiltonian.n):
                # The second index we want to capture:
                for jj in range(self.hamiltonian.n):
                    # The first sum index:
                    for kk in range(self.hamiltonian.n):
                        # The second sum index:
                        for ll in range(self.hamiltonian.n):
                            for mm, q in enumerate(np.arange(-1., 2., 1)):
                                # first term in the commutator, first part:
                                ev_mat[self.__density_index(ii, jj),
                                       self.__density_index(ll, jj)] -= \
                                d_q_star[key][mm, ll, kk]*d_q_bare[key][mm, kk, ii]
                                # first term in the commutator, second part:
                                ev_mat[self.__density_index(ii, jj),
                                       self.__density_index(kk, ll)] += \
                                d_q_star[key][mm, kk, ii]*d_q_bare[key][mm, jj, ll]

                                # second term in the commutator, first part:
                                ev_mat[self.__density_index(ii, jj),
                                       self.__density_index(ll, kk)] += \
                                d_q_star[key][mm, ll, ii]*d_q_bare[key][mm, jj, kk]
                                # second term in the commutator, second part:
                                ev_mat[self.__density_index(ii, jj),
                                       self.__density_index(ii, ll)] -= \
                                d_q_star[key][mm, jj, kk]*d_q_bare[key][mm, kk, ll]

            # Normalize:
            ev_mat = 0.5*gamma*ev_mat

            # Save the decay rates for the evolve_motion function:
            self.decay_rates[key] = -np.real(np.array(
                [ev_mat[self.__density_index(ii, ii), self.__density_index(ii, ii)]
                 for ii in range(self.hamiltonian.n)]
                ))

            # These are useful for the random evolution part:
            self.decay_rates_truncated[key] = self.decay_rates[key][self.decay_rates[key]>0]
            self.decay_rho_indices[key] = np.array([self.__density_index(ii, ii)
                for ii, rate in enumerate(self.decay_rates[key]) if rate>0]
            )
            self.recoil_velocity[key] = \
                self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['k']\
                /self.hamiltonian.mass

            self.ev_mat['decay'] += ev_mat

        return self.ev_mat['decay']

    # is only ran in setup
    def __build_transform_matrices(self):
        self.U = np.zeros((self.hamiltonian.n**2, self.hamiltonian.n**2),
                     dtype='complex128')
        self.Uinv = np.zeros((self.hamiltonian.n**2, self.hamiltonian.n**2),
                        dtype='complex128')

        for ii in range(self.hamiltonian.n):
            self.U[self.__density_index(ii, ii),
                   self.__density_index(ii, ii)] = 1.
            self.Uinv[self.__density_index(ii, ii),
                      self.__density_index(ii, ii)] = 1.

        for ii in range(self.hamiltonian.n):
            for jj in range(ii+1, self.hamiltonian.n):
                    self.U[self.__density_index(ii, jj),
                           self.__density_index(ii, jj)] = 1.
                    self.U[self.__density_index(ii, jj),
                           self.__density_index(jj, ii)] = 1j

                    self.U[self.__density_index(jj, ii),
                           self.__density_index(ii, jj)] = 1.
                    self.U[self.__density_index(jj, ii),
                           self.__density_index(jj, ii)] = -1j

        for ii in range(self.hamiltonian.n):
            for jj in range(ii+1, self.hamiltonian.n):
                    self.Uinv[self.__density_index(ii, jj),
                              self.__density_index(ii, jj)] = 0.5
                    self.Uinv[self.__density_index(ii, jj),
                              self.__density_index(jj, ii)] = 0.5

                    self.Uinv[self.__density_index(jj, ii),
                              self.__density_index(ii, jj)] = -0.5*1j
                    self.Uinv[self.__density_index(jj, ii),
                              self.__density_index(jj, ii)] = +0.5*1j

    # is only ran in setup
    def __transform_ev_matrix(self, ev_mat):
        if not hasattr(self, 'U'):
            self.__build_transform_matrices()

        ev_mat_new = self.Uinv @ ev_mat @ self.U

        # This should remove the imaginary component.
        if np.allclose(np.imag(ev_mat_new), 0):
            return np.real(ev_mat_new)
        else:
            raise ValueError('Something went dreadfully wrong.')

    # is only ran in setup
    def __transform_ev_matrices(self):
        self.ev_mat['decay'] = self.__transform_ev_matrix(self.ev_mat['decay'])
        self.ev_mat['H0'] = self.__transform_ev_matrix(self.ev_mat['H0'])

        self.ev_mat['reE'] = {}
        self.ev_mat['imE'] = {}
        for key in self.ev_mat['d_q'].keys():
            self.ev_mat['reE'][key] = np.array([self.__transform_ev_matrix(
                self.ev_mat['d_q'][key][jj] + self.ev_mat['d_q*'][key][jj]
                ) for jj in range(3)])
            # Unclear why the following works, I calculate that there should
            # be a minus sign out front.
            self.ev_mat['imE'][key] = np.array([self.__transform_ev_matrix(
                1j*(self.ev_mat['d_q'][key][jj] - self.ev_mat['d_q*'][key][jj])
                ) for jj in range(3)])

        # Transform Bq back into Bx, By, and Bz (making it real):
        self.ev_mat['B'] = spherical2cart(self.ev_mat['B'])

        self.ev_mat['B'] = np.real(np.array([
            self.__transform_ev_matrix(self.ev_mat['B'][jj]) for jj in range(3)
        ]))

        del self.ev_mat['d_q']
        del self.ev_mat['d_q*']



    def __reshape_rho(self, rho):
        rho = jnp.asarray(rho)
        if self.transform_into_re_im:
            rho = rho.astype('complex128')

            if len(rho.shape) == 1:
                rho = jnp.dot(self.U, rho)
            else:
                rho = jnp.tensordot(self.U, rho, axes=([1], [0]))

        rho = jnp.reshape(rho, (self.hamiltonian.n, self.hamiltonian.n) + rho.shape[1:])
        

        """# If not:
        if self.transform_into_re_im:
            new_rho = np.zeros(rho.shape, dtype='complex128')
            for jj in range(new_rho.shape[2]):
                new_rho[:, :, jj] = (np.diag(np.diagonal(rho[:, :, jj])) +
                                     np.triu(rho[:, :, jj], k=1) +
                                     np.triu(rho[:, :, jj], k=1).T +
                                     1j*np.tril(rho[:, :, jj], k=-1) -
                                     1j*np.tril(rho[:, :, jj], k=-1).T)
            rho = new_rho"""

        return rho


    def __reshape_sol(self):
        """
        Reshape the solution to have all the proper parts.
        """
        
        for sol in self.sols: # here all solutions is a list of atoms 
            # sol with shape (time_steps, state_dim)
            # transposed to get the same axis as original pylcp
            rho_flat = sol.y[:, -6].T
            sol.rho = self.__reshape_rho(rho_flat)
            
            sol.r = jnp.real(sol.y[:, -3:].T)
            sol.v = jnp.real(sol.y[:, -6:-3].T)

        del self.sol.y


    def set_initial_rho(self, rho0):
        """
        Sets the initial :math:`\\rho` matrix

        Parameters
        ----------
        rho0 : array_like
            The initial :math:`\\rho`.  It must have :math:`n^2` elements, where :math:`n`
            is the total number of states in the system.  If a flat array, it
            will be reshaped.
        """
        rho0 = jnp.asarray(rho0)
        if jnp.any(jnp.isnan(rho0)) or jnp.any(jnp.isinf(rho0)):
            raise ValueError('rho0 has NaNs or Infs!')

        if rho0.size != self.hamiltonian.n**2:
            raise ValueError('rho0 should have n^2 elements.')

        if rho0.shape == (self.hamiltonian.n, self.hamiltonian.n):
            rho0 = rho0.flatten()

        if self.transform_into_re_im and rho0.dtype == jnp.complex128:
            # self.rho0 = self.Uinv @ rho0
            self.rho0 = jnp.dot(self.Uinv, rho0)
        elif (not self.transform_into_re_im and rho0.dtype != jnp.complex128):
            self.rho0 = rho0.astype(jnp.complex128)
        else:
            self.rho0 = rho0

    def set_initial_rho_equally(self):
        """
        Sets the initial :math:`\\rho` matrix such that all states have the same
        population.
        """
        if self.transform_into_re_im:
            self.rho0 = jnp.zeros((self.hamiltonian.n**2,))
        else:
            self.rho0 = jnp.zeros((self.hamiltonian.n**2,), dtype=jnp.complex128)

        # only runs in initialization so is okay with for loop
        for jj in range(self.hamiltonian.ns[0]):
            self.rho0 = self.rho0.at[self.__density_index(jj, jj)].set(1/self.hamiltonian.ns[0])

    def set_initial_rho_from_populations(self, Npop):
        """
        Sets the diagonal elements of the initial :math:`\\rho` matrix

        Parameters
        ----------
        Npop : array_like
            Array of the initial populations of the states in the system.  The
            length must be :math:`n`, where :math:`n` is the number of states.
        """
        Npop = jnp.asarray(Npop)
        if len(Npop) != self.hamiltonian.n:
            raise ValueError('Npop has only %d entries for %d states.' %
                             (len(Npop), self.hamiltonian.n))
        if jnp.any(jnp.isnan(Npop)) or jnp.any(jnp.isinf(Npop)):
            raise ValueError('Npop has NaNs or Infs!')
        
        if self.transform_into_re_im:
            self.rho0 = jnp.zeros((self.hamiltonian.n**2,))
        else:
            self.rho0 = jnp.zeros((self.hamiltonian.n**2,), dtype=jnp.complex128)

        Npop = Npop/jnp.sum(Npop) # Just make sure it is normalized.
        # okay with for loop for initialization function
        for jj in range(self.hamiltonian.n):
            idx = self.__density_index(jj, jj)
            self.rho0 = self.rho0.at[idx].set(Npop[jj])

    def set_initial_rho_from_rateeq(self):
        """
        Sets the diagonal elements of the initial :math:`\\rho` matrix using
        the equilibrium populations as determined by pylcp.rateeq
        """
        # will still work since it calls `set_initial_rho_from_populations` which will transform the numpy array until `rateeq` has been implemented in jax
        if not hasattr(self, 'rateeq'):
            self.rateeq = rateeq(self.laserBeams, self.magField, self.hamiltonian)
        Neq = self.rateeq.equilibrium_populations(self.r0, self.v0, t=0)
        self.set_initial_rho_from_populations(Neq)


    def full_OBE_ev_scratch(self, r, t):
        """
        Calculate the evolution for the density matrix

        This function calculates the OBE evolution matrix at position t and r
        from scratch, first computing the full Hamiltonian, then the
        OBE evolution matrix computed via commutators, then adding in the decay
        matrix evolution.  If `Bq` is `None`, it will compute `Bq`.

        Parameters
        ----------
        r : array_like, shape (3,)
            Position at which to calculate evolution matrix
        t : float
            Time at which to calculate evolution matrix

        Returns
        -------
        ev_mat : array_like
            Evolution matrix for the densities
        """
        Eq = {}
        for key in self.laserBeams.keys():
            Eq[key] = self.laserBeams[key].total_electric_field(r, t)

        B = self.magField.Field(r, t)
        Bq = cart2spherical(B)

        H = self.hamiltonian.return_full_H(Bq, Eq)
        ev_mat = self.__build_coherent_ev_submatrix(H)

        if self.transform_into_re_im:
            return self.__transform_ev_matrix(ev_mat + self.ev_mat['decay'])
        else:
            return ev_mat + self.ev_mat['decay']


    def full_OBE_ev(self, r, t):
        """
        Calculate the evolution for the density matrix

        This function calculates the OBE evolution matrix by assembling
        pre-stored versions of the component matries.  This should be
        significantly faster than full_OBE_ev_scratch, but it may suffer bugs
        in the evolution that full_OBE_ev_scratch will not. If Bq is None,
        it will compute Bq based on r, t

        Parameters
        ----------
        r : array_like, shape (3,)
            Position at which to calculate evolution matrix
        t : float
            Time at which to calculate evolution matrix

        Returns
        -------
        ev_mat : array_like
            Evolution matrix for the densities
        """
        ev_mat = self.ev_mat['decay'] + self.ev_mat['H0']

        # Add in electric fields:
        for key in self.laserBeams.keys():
            if self.transform_into_re_im:
                Eq = self.laserBeams[key].total_electric_field(r, t)
                for ii in range(3):
                    ev_mat -= jnp.real(Eq[ii])*self.ev_mat['reE'][key][ii]
                    ev_mat -= jnp.imag(Eq[ii])*self.ev_mat['imE'][key][ii]
            else:
                Eq = self.laserBeams[key].total_electric_field(jnp.real(r), t)
                for ii in range(3):
                    ev_mat -= jnp.conjugate(Eq[ii])*self.ev_mat['d_q'][key][ii]
                    ev_mat -= Eq[ii]*self.ev_mat['d_q*'][key][ii]

        # Add in magnetic fields:
        # BUG: this code never does drhodt? and is not accessed, neither is rho which is never passed to the function.
        B = self.magField.Field(r, t)
        for ii, q in enumerate(range(-1, 2)):
            if self.transform_into_re_im:
                if np.abs(Bq[ii])>1e-10:
                    drhodt -= self.ev_mat['B'][ii]*B[ii] @ rho
            else:
                Bq = cart2spherical(B)
                if np.abs(Bq[2-ii])>1e-10:
                    drhodt -= (-1)**np.abs(q)*self.ev_mat['B'][ii]*Bq[2-ii] @ rho

        return ev_mat


    def __drhodt(self, r, t, rho):
        """
        It is MUCH more efficient to do matrix vector products and add the
        results together rather than to add the matrices together (as above)
        and then do the dot.  It is also most efficient to avoid doing useless
        math if the applied field is zero.
        """
        drhodt = jnp.dot(self.ev_mat['decay'], rho) + jnp.dot(self.ev_mat['H0'], rho)

        # Add in electric fields:
        if self.transform_into_re_im:
            for key in self.laserBeams.keys():
                Eq = self.laserBeams[key].total_electric_field(r, t)
                for ii, q in enumerate([-1., 2., 1]):
                    drhodt -= ((-1.) ** q * jnp.real(Eq[2-ii]) *
                                jnp.dot(self.ev_mat['reE'][key][ii], rho))
                    drhodt -= ((-1.) ** q * jnp.imag(Eq[2-ii]) *
                                jnp.dot(self.ev_mat['imE'][key][ii], rho))
        else:
            for key in self.laserBeams.keys():
                Eq = self.laserBeams[key].total_electric_field(jnp.real(r), t)
                for ii, q in enumerate([-1., 2., 1]):
                    drhodt -= ((-1.) ** q * Eq[2-ii] *
                                jnp.dot(self.ev_mat['d_q'][key][ii], rho))
                    drhodt -= ((-1.) ** q * jnp.conjugate(Eq[2-ii]) *
                                jnp.dot(self.ev_mat['d_q*'][key][ii], rho))

        # Add in magnetic fields:
        B = self.magField.Field(r, t)
        if self.transform_into_re_im:
            for ii in range(3):
                drhodt -= jnp.dot(self.ev_mat['B'][ii] * B[ii], rho)
        else:
            Bq = cart2spherical(B)
            for ii in range(3):
                drhodt -= jnp.dot(self.ev_mat['B'][ii] * jnp.conjugate(Bq[ii]), rho)

        return drhodt

    @functools.cached_property
    def _dydt(self):
        """
        Return the state-vector RHS for evolve_density as a stable callable.

        Stored as a cached_property so every access of self._dydt returns the
        *same Python object*, giving jax.jit a constant cache key and ensuring
        the XLA kernel is compiled only once per OBE instance.
        """
        def dydt(t, y):
            r    = y[-3:]
            v    = y[-6:-3]
            rho  = y[:-6]
            a    = jnp.zeros(3, dtype=y.dtype)
            drhodt = self.__drhodt(r, t, rho).astype(y.dtype)
            return jnp.concatenate((drhodt, a, v))
        return dydt

    def evolve_density(self, t_span, y0_batch=None, n_points=1000, **kwargs):
        """
        Evolve the density operators :math:`\\rho_{ij}` in time.

        This function integrates the optical Bloch equations to determine how
        the populations evolve in time.  Any initial velocity is kept constant
        while the atom potentially moves through the light field.  This function
        is therefore useful in determining average forces.  Any constant
        acceleration set when the OBEs were generated is ignored. It is
        analogous to rateeq.evolve_populations().

        Parameters
        ----------
        t_span : list or array_like
            A two element list or array that specify the initial and final time
            of integration.
        progress_bar : boolean
            Show a progress bar as the calculation proceeds.  Default:False
        **kwargs :
            Additional keyword arguments get passed to solve_ivp, which is
            what actually does the integration.

        Returns
        -------
        sol : OdeSolution
            Bunch object that contains the following fields:

                * t: integration times found by solve_ivp
                * rho: density matrix
                * v: atomic velocity (constant)
                * r: atomic position

            It contains other important elements, which can be discerned from
            scipy's solve_ivp documentation.
        """
        rtol = kwargs.get('rtol', 1e-5)
        atol = kwargs.get('atol', 1e-5)
        method = kwargs.get('method', 'Dopri5')

        if y0_batch is None:
            y0 = jnp.concatenate([self.rho0, self.v0, self.r0])
            y0_batch = y0[None, :]  # (1, state_dim)

        # self._dydt is a cached_property: same Python object every call on
        # this instance, so _batched_dense_trajectories JIT-compiles once and
        # reuses the kernel for every subsequent call (e.g. convergence loop).
        ts_grid, batched_ys = solve_ivp_dense(
            self._dydt, t_span, y0_batch,
            n_points=n_points, rtol=rtol, atol=atol, solver_type=method,
        )
        # batched_ys shape: (n_atoms, n_points, state_dim)

        class Bunch:
            pass
        self.sol = Bunch()
        self.sol.t = ts_grid
        self.sol.y = batched_ys
        # reconstructing the rho for output
        self.sol.rho = self.__reshape_rho(batched_ys[0, :, :self.hamiltonian.n**2].T)

        return self.sol


    def evolve_motion(self,
                      t_span,
                      y0_batch,
                      keys_batch,
                      freeze_axis=[False, False, False],
                      random_recoil=False,
                      max_scatter_probability=0.1,
                      **kwargs):
        """
        Evolve :math:`\\rho_{ij}` and the motion of the atom in time.

        This function evolves the optical Bloch equations, moving the atom
        along given the instantaneous force, for some period of time.

        Parameters
        ----------
        t_span : list or array_like
            A two element list or array that specify the initial and final time
            of integration.
        freeze_axis : list of boolean
            Freeze atomic motion along the specified axis.
            Default: [False, False, False]
        random_recoil : boolean
            Allow the atom to randomly recoil from scattering events.
            Default: False
        max_scatter_probability : float
            When undergoing random recoils, this sets the maximum time step such
            that the maximum scattering probability is less than or equal to
            this number during the next time step.  Default: 0.1
        progress_bar : boolean
            If true, show a progress bar as the calculation proceeds.
            Default: False
        record_force : boolean
            If true, record the instantaneous force and store in the solution.
            Default: False
        rng : numpy.random.Generator()
            A properly-seeded random number generator.  Default: calls
            ``numpy.random.default.rng()``
        **kwargs :
            Additional keyword arguments get passed to solve_ivp_random, which
            is what actually does the integration.

        Returns
        -------
        sol : OdeSolution
            Bunch object that contains the following fields:

                * t: integration times found by solve_ivp
                * rho: density matrix
                * v: atomic velocity
                * r: atomic position

            It contains other important elements, which can be discerned from
            scipy's solve_ivp documentation.
        """
        free_axes = jnp.bitwise_not(jnp.asarray(freeze_axis, dtype=bool))
        random_recoil_flag = random_recoil

        def dydt(t, y):
            # since jax handles batching, y is 1D array of one atom
            r = y[-3:]
            v = y[-6 : -3]
            rho = y[:-6]
            
            F = self.force(r, t, rho, return_details=False)
            
            dvdt = (F * free_axes) / self.hamiltonian.mass + self.constant_accel
            drdt = v
            drhodt = self.__drhodt(r, t, rho)

            return jnp.concatenate((drhodt, dvdt, drdt))
        
        def _jax_random_vector(key):
            key_phi, key_z = jax.random.split(key)
            phi = 2.0 * jnp.pi * jax.random.uniform(key_phi)
            z = 2.0 * jax.random.uniform(key_z) - 1.0
            
            r_vec = jnp.sqrt(1.0 - z**2)
            return jnp.array([r_vec * jnp.cos(phi), r_vec * jnp.sin(phi), z]) * free_axes
        
        def random_recoil_fn(t, y, dt, key):
            num_of_scatters = 0
            total_P = 0.
            
            y_jump = y
            
            for decay_key in self.decay_rates:
                P = dt * self.decay_rates_truncated[decay_key] * jnp.real(y[self.decay_rho_indices[decay_key]])
                
                key, subkey_dice, subkey_v1, subkey_v2 = jax.random.split(key, 4)
                dice = jax.random.uniform(subkey_dice, shape=P.shape)
                scatters_mask = jnp.where(dice < P, 1, 0)
                num_scatters_this_channel = jnp.sum(scatters_mask)
                
                vec1 = _jax_random_vector(subkey_v1)
                vec2 = _jax_random_vector(subkey_v2)
                
                kick = self.recoil_velocity[decay_key] * (vec1 + vec2)
                
                y_jump = jnp.where(
                    num_scatters_this_channel > 0,
                    y_jump.at[-6:-3].add(kick * num_scatters_this_channel),
                    y_jump
                )
                
                num_of_scatters += num_scatters_this_channel
                total_P += jnp.sum(P)
            
            new_dt_max = jnp.where(total_P > 0, (max_scatter_probability / total_P) * dt, jnp.inf)

            return y_jump, num_of_scatters, new_dt_max, key
        

        if not random_recoil_flag:
            def dummy_recoil(t, y, dt, key):
                return y, 0, jnp.inf, key
                
            self.sols = solve_ivp_random(
                fun=dydt,
                random_func=dummy_recoil,
                t_span=t_span,
                y0_batch=y0_batch,
                keys_batch=keys_batch, 
                **kwargs
            )
            
        else:
            self.sols = solve_ivp_random(
                fun=dydt,
                random_func=random_recoil_fn,
                t_span=t_span,
                y0_batch=y0_batch,
                keys_batch=keys_batch,
                **kwargs
            )


        # Remake the solution:
        self.__reshape_sol()


        return self.sols


    def observable(self, O, rho=None):
        """
        Calculates the expectation value of the observable O given density matrix rho.

        This method computes the trace of the product of the observable operator
        and the density matrix. It natively supports evaluating single states 
        as well as batched/time-series density matrices using JAX.

        Parameters
        ----------
        O : array_like
            The matrix form of the observable operator. Can have any shape,
            representing scalar, vector, or tensor operators, but the last two
            axes must correspond to the matrix of the operator and have the
            same dimensions as the generating Hamiltonian. For example,
            a vector operator might have the shape (3, n, n), where n
            is the number of states and the first axis corresponds to x, y,
            and z.
        rho : array_like, optional
            The density matrix. The first two dimensions must have sizes
            (n, n), but there may be multiple instances of the density matrix
            tiled in higher dimensions (e.g., (n, n, m) for a time series). 
            Alternatively, a flat 1D array of length n**2 can be provided 
            and will be automatically reshaped. If not specified, the method 
            will default to the current solution stored in memory (`self.sols` 
            or `self.sol`).

        Returns
        -------
        observable : jax.Array
            The calculated expectation value(s). The output shape is 
            `O.shape[:-2] + rho.shape[2:]`.
        """
        if rho is None: # handle if rho is missing
            if hasattr(self, 'sols') and isinstance(self.sols, list) and len(self.sols) > 0:
                rho = self.sols[0].rho
            elif hasattr(self, 'sol'):
                rho = self.sol.rho
            else:
                raise ValueError("No solution found in memory. Please provide 'rho' explicitly.")
        
        O = jnp.asarray(O) # convert to jax array in case
        if rho.shape[0] == self.hamiltonian.n**2 and rho.ndim == 1:
            rho_mat = self.__reshape_rho(rho)
        else:
            rho_mat = jnp.asarray(rho)

        if rho.shape[:2]!=(self.hamiltonian.n, self.hamiltonian.n):
            raise ValueError('rho must have dimensions (n, n,...), where n '+
                             'corresponds to the number of states in the '+
                             'generating Hamiltonian. ' +
                             'Instead, shape of rho is %s.'%str(rho.shape))
        elif O.shape[-2:]!=(self.hamiltonian.n, self.hamiltonian.n):
            raise ValueError('O must have dimensions (..., n, n), where n '+
                             'corresponds to the number of states in the '+
                             'generating Hamiltonian. ' +
                             'Instead, shape of O is %s.'%str(O.shape))
        av0 = jnp.tensordot(O, rho_mat, axes=[(-2, -1), (0, 1)])
        if jnp.allclose(jnp.imag(av0), jnp.zeros_like(jnp.imag(av0))):
            av0 = jnp.real(av0)
        return av0


    def force(self, r, t, rho, return_details=False):
        """
        Calculates the instantaneous force on the atom.

        This method computes the gradient of the laser and magnetic fields 
        and traces them with the density matrix to find the instantaneous 
        expectation value of the force. It supports evaluating single points 
        as well as batched/time-series data arrays natively using JAX.

        Parameters
        ----------
        r : array_like, shape (3,) or (3, n_pts)
            Position(s) at which to calculate the force.
        t : float or array_like, shape (n_pts,)
            Time(s) at which to calculate the force.
        rho : array_like
            Density matrix (or flat state vector representation) with which to 
            calculate the force.
        return_details : boolean, optional
            If true, returns the detailed components of the force broken down 
            by laser, polarization, and magnetic fields. Default: False.

        Returns
        -------
        f : jax.Array
            Total instantaneous force experienced by the atom.
        f_laser : dict of jax.Array
            (Returned if `return_details=True`) The forces due to each laser, 
            indexed by the manifold the laser addresses.
        f_laser_q : dict of jax.Array
            (Returned if `return_details=True`) The forces due to each laser 
            and its q component of the polarization, indexed by the manifold 
            the laser addresses.
        f_mag : jax.Array
            (Returned if `return_details=True`) The forces due to the 
            magnetic field.
        """
        
        rho_mat = self.__reshape_rho(rho)

        f = jnp.zeros((3,) + rho_mat.shape[2:])
        f_mag = jnp.zeros_like(f)
        if return_details:
            f_laser_q = {}
            f_laser = {}
            

        # Helper: evaluate field gradient at r (3,) or (3, n_pts) and t scalar/(n_pts,)
        def _grad(beam_func, r_in, t_in):
            if jnp.asarray(r_in).ndim == 2:  # time series
                t_arr = jnp.asarray(t_in).ravel()
                dE = jax.vmap(lambda ri, ti: beam_func(ri, ti))(
                    jnp.real(r_in).T, t_arr  # (n_pts, 3)
                )
                return jnp.moveaxis(dE, 0, -1)  # (3, 3, n_pts)
            else:
                return beam_func(jnp.real(r_in), t_in)

        for key in self.laserBeams:
            # First, determine the average mu_q:
            # This returns a (3,) + rho.shape[2:] array
            mu_q_av = self.observable(self.hamiltonian.d_q_bare[key], rho_mat)
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['gamma']

            if not return_details:
                delE = _grad(self.laserBeams[key].total_electric_field_gradient, r, t)

                # We are just looking at the d_q, whereas the full observable
                # is \nabla (d_q \cdot E^\dagger) + (d_q^* E)) =
                # 2 Re[\nabla (d_q\cdot E^\dagger)].  Putting in the units,
                # we see we need a factor of gamma/4, making
                # this 2 Re[\nabla (d_q\cdot E^\dagger)]/4 =
                # Re[\nabla (d_q\cdot E^\dagger)]/2
                for jj, q in enumerate([-1., 2., 1.]):
                    f += jnp.real((-1) ** q * gamma * mu_q_av[jj] * delE[:, 2-jj])/2
            else:
                num_beams = self.laserBeams[key].num_of_beams

                f_laser_q_key = jnp.zeros((3, 3, num_beams) + rho_mat.shape[2:])
                f_laser_key = jnp.zeros((3, num_beams) + rho_mat.shape[2:])

                # Now, dot it into each laser beam:
                for ii, beam in enumerate(self.laserBeams[key].beam_vector):
                    delE = _grad(beam.electric_field_gradient, r, t)

                    for jj, q in enumerate([-1., 2., 1.]):
                        val = jnp.real((-1) ** q * gamma * mu_q_av[jj] * delE[:, 2-jj])/2
                        f_laser_q_key = f_laser_q_key.at[:, jj, ii].add(val)

                    f_laser_key = f_laser_key.at[:, ii].set(jnp.sum(f_laser_q_key[:, :, ii], axis=1))

                f = f + jnp.sum(f_laser_key, axis=1)

                f_laser_q[key] = f_laser_q_key
                f_laser[key] = f_laser_key
                
        # Are we including magnetic forces?
        if self.include_mag_forces:
            av_mu = self.observable(self.hamiltonian.mu, rho_mat)

            if rho_mat.ndim == 3:  # time series: r shape (3, n_pts)
                delB = jax.vmap(lambda ri: self.magField.gradField(ri))(jnp.real(r).T)
                # delB: (n_pts, 3, 3) -> vmap over time, contract i
                f_mag = jnp.real(jnp.einsum('it,tij->jt', av_mu, delB))
            else:
                delB = self.magField.gradField(jnp.real(r))
                f_mag = jnp.real(jnp.einsum('...i,...ij->...j', av_mu, delB))

            f += f_mag

        if return_details:
            return f, f_laser, f_laser_q, f_mag
        else:
            return f


    def find_equilibrium_force(self, deltat=500, itermax=100, Npts=5001,
                               rel=1e-5, abs=1e-9, debug=False,
                               initial_rho='rateeq',
                               return_details=False, **kwargs):
        """
        Finds the equilibrium force at the initial position.

        This method works by solving the OBEs in a chunk of time
        $\Delta T$, calculating the force during that chunk, continuing
        the integration for another chunk, calculating the force during that
        subsequent chunk, and comparing the average of the forces of the two
        chunks to see if they have converged.

        Parameters
        ----------
        deltat : float, optional
            Chunk time $\Delta T$ to integrate over before checking for convergence. Default: 500.
        itermax : int, optional
            Maximum number of chunk iterations. Default: 100.
        Npts : int, optional
            Number of points to divide the chunk into. Default: 5001.
        rel : float, optional
            Relative convergence parameter. Default: 1e-5.
        abs : float, optional
            Absolute convergence parameter. Default: 1e-9.
        debug : boolean, optional
            If true, prints out debug information and stores piecewise solutions 
            in `self.piecewise_sols` as the integration proceeds. Default: False.
        initial_rho : {'rateeq', 'equally', 'frompops'}, optional
            Determines how to set the initial density matrix $\\rho$ at the start
            of the calculation. Default: 'rateeq'.
        return_details : boolean, optional
            If true, returns the detailed components of the force (laser, magnetic, 
            polarization projections) and equilibrium populations. Default: False.
            
        Other Parameters
        ----------------
        init_pop : array_like, optional
            Initial populations to use if `initial_rho='frompops'`.
        **kwargs : 
            Additional keyword arguments are passed directly to `evolve_density`
            (e.g., `rtol` and `atol` for the ODE solver).

        Returns
        -------
        F : jax.Array, shape (3,)
            Total equilibrium force experienced by the atom.
        F_laser : dict of jax.Array
            (Returned if `return_details=True`) The forces due to each laser, indexed
            by the manifold the laser addresses.
        F_laser_q : dict of jax.Array
            (Returned if `return_details=True`) The forces due to each laser and its 
            q component, indexed by the manifold the laser addresses.
        F_mag : jax.Array, shape (3,)
            (Returned if `return_details=True`) The forces due to the magnetic field.
        Neq : jax.Array
            (Returned if `return_details=True`) The equilibrium populations.
        ii : int
            (Returned if `return_details=True`) Number of iterations needed to converge.
        """
        if initial_rho == 'rateeq':
            self.set_initial_rho_from_rateeq()
        elif initial_rho == 'equally':
            self.set_initial_rho_equally()
        elif initial_rho == 'frompops':
            Npop = kwargs.pop('init_pop', None)
            self.set_initial_rho_from_populations(Npop)
        else:
            raise ValueError('Argument initial_rho=%s not understood' % initial_rho)

        old_f_avg = jnp.full((3,), jnp.inf)

        if debug:
            print('Finding equilibrium force at ' +
                  'r=(%.2f, %.2f, %.2f) ' % (self.r0[0], self.r0[1], self.r0[2]) +
                  'v=(%.2f, %.2f, %.2f) ' % (self.v0[0], self.v0[1], self.v0[2]) +
                  'with deltat = %.2f, itermax = %d, Npts = %d, ' % (deltat, itermax, Npts) +
                  'rel = %.1e and abs = %.1e' % (rel, abs))
            self.piecewise_sols = []

        ii = 0
        while True:
            # Build single-atom state vector [rho_flat, v, r] and wrap as batch
            y0 = jnp.concatenate([self.rho0, self.v0, self.r0])
            y0_batch = y0[None, :]  # shape (1, state_dim)

            self.evolve_density(
                [ii * deltat, (ii + 1) * deltat],
                y0_batch,
                n_points=int(Npts),
                **kwargs
            )

            # Extract atom 0: self.sol.y shape (1, n_points, state_dim)
            y_atom = self.sol.y[0]  # (n_points, state_dim)
            rho_flat = y_atom[:, :-6].T  # (n^2, n_points)
            r = jnp.real(y_atom[:, -3:]).T  # (3, n_points)

            f, f_laser, f_laser_q, f_mag = self.force(
                r, self.sol.t, rho_flat, return_details=True
            )
            f_avg = jnp.mean(f, axis=1)  # (3,)

            if debug:
                print(ii, f_avg, jnp.sum(f_avg ** 2))
                self.piecewise_sols.append(self.sol)

            f_sq = jnp.sum(f_avg ** 2)
            diff_sq = jnp.sum((old_f_avg - f_avg) ** 2)
            converged = bool(
                (f_sq < abs) or
                (diff_sq / jnp.maximum(f_sq, 1e-30) < rel) or
                (diff_sq < abs)
            )
            if converged or ii >= itermax - 1:
                break

            old_f_avg = f_avg
            # Seed next chunk: rho_flat[:, -1] is already in the correct flat
            # representation (re/im if transform_into_re_im, complex otherwise)
            self.set_initial_rho(rho_flat[:, -1])
            self.set_initial_position_and_velocity(r[:, -1], jnp.real(y_atom[-1, -6:-3]))
            ii += 1

        if return_details:
            f_laser_avg = {key: jnp.mean(f_laser[key], axis=2) for key in f_laser}
            f_laser_avg_q = {key: jnp.mean(f_laser_q[key], axis=3) for key in f_laser_q}
            f_mag_avg = jnp.mean(f_mag, axis=1)
            rho_mat = self.__reshape_rho(rho_flat)  # (n, n, n_points)
            Neq = jnp.real(jnp.diagonal(jnp.mean(rho_mat, axis=2)))
            return (f_avg, f_laser_avg, f_laser_avg_q, f_mag_avg, Neq, ii)
        else:
            return f_avg


    def generate_force_profile(self, R, V, name=None, **kwargs):
        """
        Map out the equilibrium force vs. position and velocity using batched JAX integration.

        This method solves the Optical Bloch Equations (OBEs) simultaneously across a 
        grid of initial positions and velocities. It integrates the evolution in chunks 
        of time, comparing the time-averaged force of successive chunks until the force 
        has converged for all grid points.

        Parameters
        ----------
        R : array_like, shape(3, ...)
            Position vector. First dimension of the array must be length 3, and
            corresponds to :math:`x`, :math:`y`, and :math:`z` components,
            respectively.
        V : array_like, shape(3, ...)
            Velocity vector. First dimension of the array must be length 3, and
            corresponds to :math:`v_x`, :math:`v_y`, and :math:`v_z` components,
            respectively.
        name : str, optional
            Name for the profile. Stored in the profile dictionary in this object.
            If None, uses the next integer, cast as a string, (i.e., '0') as
            the name.
            
        Other Parameters
        ----------------
        deltat : float, optional
            Chunk time :math:`\\Delta T` to integrate over before checking for convergence. 
            Default: 500.
        itermax : int, optional
            Maximum number of chunk iterations to perform. Default: 100.
        Npts : int, optional
            Number of points to divide the integration chunk into. Default: 5001.
        rel : float, optional
            Relative convergence parameter. Default: 1e-5.
        abs : float, optional
            Absolute convergence parameter. Default: 1e-9.
        initial_rho : {'rateeq', 'equally'}, optional
            Determines how to set the initial density matrix :math:`\\rho` at the start 
            of the calculation. Default: 'rateeq'.
        deltat_r : float, optional
            Dynamic deltat scaling factor based on spatial position. 
        deltat_v : float, optional
            Dynamic deltat scaling factor based on velocity.
        deltat_tmax : float, optional
            Maximum allowed deltat if dynamically calculated via `deltat_r` or `deltat_v`.
            Default: np.inf.
        deltat_func : callable, optional
            A custom function `f(r, v)` to dynamically determine the chunk time `deltat` 
            for each grid point. The method will use the minimum valid deltat returned 
            across the grid to ensure alignment.

        Returns
        -------
        profile : pylcp.obe.force_profile
            Resulting force profile containing the equilibrium forces, detailed laser/mag 
            forces, and equilibrium populations for the specified grid.
        """
        # Pop deltat-shaping kwargs
        deltat_r    = kwargs.pop('deltat_r',    None)
        deltat_v    = kwargs.pop('deltat_v',    None)
        deltat_tmax = kwargs.pop('deltat_tmax', np.inf)
        deltat_func = kwargs.pop('deltat_func', None)
        kwargs.pop('return_details', None)  # always True here

        # Pop find_equilibrium_force kwargs so they don't leak into evolve_density
        chunk_deltat = kwargs.pop('deltat',       500)
        itermax      = kwargs.pop('itermax',      100)
        Npts         = kwargs.pop('Npts',         5001)
        rel          = kwargs.pop('rel',          1e-5)
        abs_tol      = kwargs.pop('abs',          1e-9)
        initial_rho  = kwargs.pop('initial_rho',  'rateeq')

        if not name:
            name = '{0:d}'.format(len(self.profile))

        self.profile[name] = force_profile(R, V, self.laserBeams, self.hamiltonian)

        # Flatten the position/velocity grid: (3, N)
        R_np = np.array(R).reshape(3, -1)
        V_np = np.array(V).reshape(3, -1)
        N    = R_np.shape[1]

        # Determine a single chunk_deltat for all atoms (use minimum)
        if deltat_func is not None:
            deltats = [deltat_func(R_np[:, i], V_np[:, i]) for i in range(N)]
            valid   = [float(d) for d in deltats if d is not None]
            if valid:
                chunk_deltat = min(valid)
        elif deltat_v is not None or deltat_r is not None:
            deltats = []
            for i in range(N):
                r_i, v_i = R_np[:, i], V_np[:, i]
                d = None
                if deltat_v is not None:
                    vabs = np.sqrt(np.sum(v_i**2))
                    d = float(deltat_tmax) if vabs == 0 else min(2*np.pi*deltat_v/vabs, float(deltat_tmax))
                if deltat_r is not None:
                    rabs = np.sqrt(np.sum(r_i**2))
                    d_r  = float(deltat_tmax) if rabs == 0 else min(2*np.pi*deltat_r/rabs, float(deltat_tmax))
                    d    = d_r if d is None else min(d, d_r)
                if d is not None:
                    deltats.append(d)
            if deltats:
                chunk_deltat = min(deltats)

        # Build initial rho for every atom
        rho0_list = []
        for i in range(N):
            self.set_initial_position_and_velocity(R_np[:, i], V_np[:, i])
            if initial_rho == 'rateeq':
                self.set_initial_rho_from_rateeq()
            elif initial_rho == 'equally':
                self.set_initial_rho_equally()
            else:
                raise ValueError(
                    f'initial_rho={initial_rho!r} not supported in generate_force_profile'
                )
            rho0_list.append(self.rho0)

        rho0_batch = jnp.stack(rho0_list)                    # (N, n²)
        V_jnp      = jnp.asarray(V_np.T)                     # (N, 3)
        R_jnp      = jnp.asarray(R_np.T)                     # (N, 3)
        y0_batch   = jnp.concatenate(
            [rho0_batch, V_jnp, R_jnp], axis=1
        )                                                     # (N, state_dim)

        old_f_avg = jnp.full((N, 3), jnp.inf)

        ii = 0
        while True:
            self.evolve_density(
                [ii * chunk_deltat, (ii + 1) * chunk_deltat],
                y0_batch,
                n_points=int(Npts),
                **kwargs
            )
            t = self.sol.t  # (n_pts,)

            # self.sol.y: (N, n_pts, state_dim)
            rho_flat_all = self.sol.y[:, :, :-6].transpose(0, 2, 1)    # (N, n², n_pts)
            r_all        = jnp.real(self.sol.y[:, :, -3:]).transpose(0, 2, 1)  # (N, 3, n_pts)

            # Compute forces for all atoms in parallel
            f_all, f_laser_all, f_laser_q_all, f_mag_all = jax.vmap(
                lambda r_i, rho_i: self.force(r_i, t, rho_i, return_details=True)
            )(r_all, rho_flat_all)
            # f_all: (N, 3, n_pts);  f_laser_all: {key: (N, 3, beams, n_pts)}
            # f_laser_q_all: {key: (N, 3, 3, beams, n_pts)};  f_mag_all: (N, 3, n_pts)

            f_avg = jnp.mean(f_all, axis=2)  # (N, 3)

            f_sq    = jnp.sum(f_avg ** 2, axis=1)                        # (N,)
            diff_sq = jnp.sum((old_f_avg - f_avg) ** 2, axis=1)          # (N,)
            all_converged = bool(jnp.all(
                (f_sq < abs_tol)
                | (diff_sq / jnp.maximum(f_sq, 1e-30) < rel)
                | (diff_sq < abs_tol)
            ))

            if all_converged or ii >= itermax - 1:
                break

            old_f_avg = f_avg
            # Seed next chunk directly from the last time point
            y0_batch = self.sol.y[:, -1, :]  # (N, state_dim)
            ii += 1

        # Time-average all detailed quantities
        f_laser_avg   = {key: jnp.mean(f_laser_all[key],   axis=3) for key in f_laser_all}   # (N,3,beams)
        f_laser_avg_q = {key: jnp.mean(f_laser_q_all[key], axis=4) for key in f_laser_q_all} # (N,3,3,beams)
        f_mag_avg     = jnp.mean(f_mag_all, axis=2)                                           # (N,3)

        # Equilibrium populations: mean rho over time → diagonal
        rho_flat_mean = jnp.mean(rho_flat_all, axis=2)   # (N, n²)
        def get_Neq(rho_flat_i):
            return jnp.real(jnp.diagonal(self.__reshape_rho(rho_flat_i)))
        Neq_all = jax.vmap(get_Neq)(rho_flat_mean)        # (N, n)

        # Write results back into the force_profile using nditer for multi-index
        it = np.nditer(
            [R[0], R[1], R[2], V[0], V[1], V[2]],
            flags=['refs_ok', 'multi_index'],
            op_flags=[['readonly']] * 6
        )
        for atom_idx, _ in enumerate(it):
            mi = it.multi_index
            self.profile[name].store_data(
                mi,
                Neq_all[atom_idx],
                f_avg[atom_idx],
                {key: f_laser_avg[key][atom_idx]   for key in f_laser_avg},
                f_mag_avg[atom_idx],
                ii,
                {key: f_laser_avg_q[key][atom_idx] for key in f_laser_avg_q},
            )

        return self.profile[name]
"""Nuclear potential module.

Module containing representations of 3D potentials for use in nuclear theory.
Also contains logic to read them from and save them to files with a standard
naming convention.

class PotentialType
-------------------
A container class to hold all the physical information about the potential. It
has the following method::

    potential_type = PotentialType(n_body, order, name, channel, particles)

class Potential
---------------
Abstraction for the representation of a potential. Handles the logic of adding
and removing weights. Can generate corresponding kinetic energy. It has the
following methods::

    potential = Potential(potential_type, nodes, weights, potential, lam=50.0,
                          has_weights=False)
    kinetic_energy = potential.kinetic_energy()
    potential_data_wo_weights = potential.without_weights()
    potential_data_w_weights = potential.with_weights()

Methods
-------
potential = load(file_str)

Method to load a potential from a file. Requires that standard file-naming
conventions have been followed.

save(directory, potential)

Method to save potential with correct naming convention to directory.

Changelog:

2018.11.06
    Added:
        Initial creation of module

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re

import numpy as np

NBODY_DICT = {
    'NN': 2,
    '3N': 3,
}

INV_NBODY_DICT = {v: k for k, v in NBODY_DICT.items()}

ORDER_DICT = {
    'LO': 0,
    'NLO': 1,
    'N2LO': 2,
    'N3LO': 3,
}

INV_ORDER_DICT = {v: k for k, v in ORDER_DICT.items()}


class PotentialType:
    """Container for information related to potential."""

    # pylint: disable=too-many-arguments
    def __init__(self, n_body, order, name, channel, particles):
        """Construct potential type.

        Parameters
        ----------
        n_body : int
            Number of particles interacting in potential.
        order : int
            Order to which potential was calculated.
        name : str
            Name for potential, may reflect something about origin.
        channel: (int, int, int, int, int)
            5-tuple of integers representing the partial wave channel for the
            potential.
        particles: str
            String representing constituent particles in the interaction.

        """
        self._n_body = n_body
        self._order = order
        self._name = name
        self._channel = channel
        self._particles = particles

    @property
    def n_body(self):
        """Return number of particles in potential.

        Returns
        -------
        int
            Number of particles.

        """
        return self._n_body

    @property
    def order(self):
        """Return order to which potential was calculated.

        Returns
        -------
        int
            Order of potential.

        """
        return self._order

    @property
    def name(self):
        """Return name of potential.

        Returns
        -------
        str
            Name of potential.

        """
        return self._name

    @property
    def channel(self):
        """Return partial wave channel of potential.

        Returns
        -------
        (int, int, int, int, int)
            5-tuple representation of partial wave channel.

        """
        return self._channel

    @property
    def particles(self):
        """Return particles in interaction to which potential applies.

        Returns
        -------
        str
            String with particles in interaction.

        """
        return self._particles


class Potential:
    """Class encapsulating all relevant information about a potential."""

    # pylint: disable=too-many-arguments
    def __init__(self, potential_type, nodes, weights, potential, lam=50.0,
                 has_weights=False):
        """Create potential from parameters.

        Parameters
        ----------
        potential_type : PotentialType
            PotentialType instance with information about the potential.
        nodes : list of floats
            List of momenta at which the potential is defined.
        weights : list of floats
            List of integration weights corresponding to nodes.
        potential : matrix of floats
            Value of potential at incoming and outgoing momenta in nodes.
        lam : float, optional
            Value of lambda (SRG flow parameter) for potential. For unevolved
            potentials, a value of 50 is the default.
        has_weights : bool, optional
            Specifies whether potential given has weights factored in already.

        """
        self._potential_type = potential_type
        self._nodes = nodes
        self._weights = weights
        self._lam = lam
        if has_weights:
            potential = _rem_w(potential, self._weights)
        self._potential = potential

    def with_weights(self):
        """Return potential with weights factored in (for calculations).

        Returns
        -------
        matrix of floats
            Potential with integration weights.

        """
        return _add_w(self._potential, self._weights)

    def without_weights(self):
        """Return potential without weights (for visualization).

        Returns
        -------
        matrix of floats
            Potential without integration weights.

        """
        return np.array(self._potential)

    def kinetic_energy(self):
        """Return kinetic energy for potential (for calculations).

        Returns
        -------
        matrix of floats
            Kinetic energy matrix.

        """
        return np.diag(np.array([p**2 for p in self._nodes]))

    @property
    def potential_type(self):
        """Return `PotentialType` object for potential.

        Returns
        -------
        PotentialType
            Object with all physics related information for the potential.

        """
        return self._potential_type

    @property
    def nodes(self):
        """Return the nodes for the potential.

        Returns
        -------
        list of floats
            List of momenta at which potential is defined.

        """
        return self._nodes

    @property
    def weights(self):
        """Return weights for the potential.

        Returns
        -------
        list of floats
            Integration weights corresponding to nodes for potential.

        """
        return self._weights

    @property
    def lam(self):
        """Return lambda for potential.

        Returns
        -------
        float
            Value of lambda, the SRG flow parameter, for potential.

        """
        return self._lam


# pylint: disable=too-many-locals
def load(file_str):
    """Load potential from file.

    Parameters
    ----------
    file_str : str
        String path to file with potential data.

    Returns
    -------
    Potential
        Potential created from extracted information and data from file.

    """
    # Parse info about potential from filename
    # Strip directory structure
    end = file_str.split('/')[-1]

    # Match regular expression
    regex_str = r'V(.*)_(.*)_(.*)_SLLJT_(.*)_lambda_(.*)_Np_(.*)_(.*)\.dat'
    result = re.search(regex_str, end)

    # Extract values from matches
    n_body_str = result.group(1)
    order_str = result.group(2)
    name = result.group(3)
    channel_str = result.group(4)
    lam = float(result.group(5))
    particles = result.group(7)

    # Convert string values to integer values
    n_body = NBODY_DICT[n_body_str]
    order = ORDER_DICT[order_str]

    # Conver channel to 5-tuple
    channel = tuple([int(n) for n in channel_str])

    # Get number of points
    num_points = int(result.group(6))

    # Read potential
    with open(file_str) as file:
        nodes = []
        weights = []
        for _ in range(num_points):
            vals = file.readline().split()
            weights.append(float(vals[0]))
            nodes.append(float(vals[1]))
        potential = np.array([[float(file.readline().split()[-1]) for _ in
                               range(num_points)] for _ in range(num_points)])

    # Create potential_type
    potential_type = PotentialType(n_body, order, name, channel, particles)

    # Return potential
    return Potential(potential_type, nodes, weights, potential, lam)


def save(dir_str, potential):
    """Save potential to directory with correct file-naming.

    Parameters
    ----------
    dir_str : str
        String corresponding to directory where file should be saved. May have
        trailing `/`.
    potential : Potential
        Potential to be saved.

    """
    # Strip potential trailing '/'
    if dir_str[-1] == '/':
        dir_str = dir_str[:-1]

    # Set up format strings
    file_format_str = '{}/V{}_{}_{}_SLLJT_{}_lambda_{:.2}_Np_{}_{}.dat'
    nodes_format_str = '{:.5e} {:.5e}\n'
    potential_format_str = '{:.5e} {:.5e} {:.5e}\n'

    # Get values for format string
    potential_type = potential.potential_type
    n_body = potential_type.n_body
    n_body_str = INV_NBODY_DICT[n_body]
    order = potential_type.order
    order_str = INV_ORDER_DICT[order]
    name = potential_type.name
    channel_str = ''.join(list(potential_type.channel))
    lam = potential_type.lam
    num_points = len(potential.nodes)
    particles = potential_type.particles

    # Create full file path string
    file_path = file_format_str.format(n_body_str, order_str, name,
                                       channel_str, lam, num_points, particles)

    # Create directory if it doesnt exist
    _ensure_dir_for_file(file_path)

    # Output potential
    with open(file_path, 'w+') as file:
        for weight, node in zip(potential.weights, potential.nodes):
            file.write(nodes_format_str.format(weight, node))
        for i in range(num_points):
            for j in range(num_points):
                file.write(potential_format_str.format(
                    potential.nodes[i], potential.nodes[j],
                    potential.without_weights()[i][j]))


# ------------------- Internal Methods ------------------------------------- #


def _add_w(matrix, weights):
    factor_vector = [x**0.5 for x in weights]
    weighted_matrix = np.dot(np.dot(np.diag(factor_vector), matrix),
                             np.diag(factor_vector))
    return weighted_matrix


def _rem_w(matrix, weights):
    factor_vector = [1/(x**0.5) for x in weights]
    unweighted_matrix = np.dot(np.dot(np.diag(factor_vector), matrix),
                               np.diag(factor_vector))
    return unweighted_matrix


def _ensure_dir_for_file(file):
    directory = os.path.dirname(file)
    if not os.path.exists(directory):
        os.makedirs(directory)

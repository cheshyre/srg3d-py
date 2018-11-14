"""Library for the easy evolution of momentum-space potentials in 3D.

Provides:
    1. An SRG object to evolve potentials
    2. Classes for single channel and coupled channel potentials
    3. Interface to read potentials from files and save them in standard
        directories

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from srg3d import potential
from srg3d import srg

__all__ = ['srg', 'potential']

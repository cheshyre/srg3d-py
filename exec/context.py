"""Set python path to import package properly."""
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

# pylint: disable=wrong-import-position
import srg3d    # noqa: E402

__all__ = ['srg3d']

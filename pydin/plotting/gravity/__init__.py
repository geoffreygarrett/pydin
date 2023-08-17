# plotting/gravity/__init__.py

from .. import attempt_import, requires

# Attempt to import your dependencies
matplotlib = attempt_import('matplotlib')
mayavi = attempt_import('mayavi.mlab')

# Import your functions conditionally:

if matplotlib is not None:
    pass
else:
    @requires("matplotlib")
    def plot_gravity_2d(*args, **kwargs):
        """ This function requires 'matplotlib'. """
        pass


    @requires("matplotlib")
    def plot_gravity_3d(*args, **kwargs):
        """ This function requires 'matplotlib'. """
        pass

if mayavi is not None:
    from ._mayavi import *

else:
    @requires("mayavi.mlab")
    def plot_gravity_3d_mayavi(*args, **kwargs):
        """ This function requires 'mayavi.mlab'. """
        pass


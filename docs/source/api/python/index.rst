Python API
==========

The preferred Python entry points are the public facade modules. Native binding
classes may appear in signatures because the facade delegates to the compiled
extension, but user code should import through ``mtlearn.morphology`` and
``mtlearn.layers``.

.. toctree::
   :maxdepth: 2

   morphology
   layers
   native
   data
   datasets

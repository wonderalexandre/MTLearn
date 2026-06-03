Layers
======

The primary trainable preprocessing layer is
``ConnectedFilterPreprocessingLayer``. Reference and legacy implementations are
documented as compatibility surfaces, but new experiments should prefer the
primary layer.

Primary CFP Layer
-----------------

.. autoclass:: mtlearn.layers.ConnectedFilterPreprocessingLayer
   :members:
   :show-inheritance:

.. autoclass:: mtlearn.layers.CFPValuation
   :members:
   :show-inheritance:

Checkpoint Helpers
------------------

.. autofunction:: mtlearn.layers.collect_cfp_configs

.. autofunction:: mtlearn.layers.save_checkpoint

.. autofunction:: mtlearn.layers.load_checkpoint

Reference Implementations
-------------------------

.. autoclass:: mtlearn.layers.ConnectedFilterPreprocessingLayerLegacy
   :members:
   :show-inheritance:

.. autoclass:: mtlearn.layers.ConnectedFilterPreprocessingLayerWithExplicitJacobian
   :members:
   :show-inheritance:

.. autoclass:: mtlearn.layers.ConnectedFilterPreprocessingLayerWithCPUTreeTraversal
   :members:
   :show-inheritance:

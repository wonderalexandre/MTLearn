CFP Preparation and Storage
===========================

Import these public components from ``mtlearn.layers.cfp``. They separate fixed
CPU morphology from learned parameters, normalization and optional retention.
For policy choices and executable patterns, see
:doc:`../../guides/cfp-cache-migration`. Layer methods such as ``fit_stats``,
``set_stats`` and ``forward_prepared`` are documented in :doc:`layers`.

Preparation
-----------

.. autoclass:: mtlearn.layers.cfp.CFPPreprocessor
   :members:

.. autoclass:: mtlearn.layers.cfp.PreparedMorphology
   :members:

.. autoclass:: mtlearn.layers.cfp.PreparedBatch
   :members:

.. autoclass:: mtlearn.layers.cfp.PreparationResult
   :members:

.. autoclass:: mtlearn.layers.cfp.PreparedDataset
   :members:

.. autofunction:: mtlearn.layers.cfp.collate_prepared

Storage policies
----------------

.. autoclass:: mtlearn.layers.cfp.NullStore
   :members:

.. autoclass:: mtlearn.layers.cfp.MemoryStore
   :members:

.. autoclass:: mtlearn.layers.cfp.DiskStore
   :members:

Normalization snapshot
----------------------

.. autoclass:: mtlearn.layers.cfp.StatisticsSnapshot
   :members:

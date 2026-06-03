Native Binding Classes
======================

These classes are exposed by the compiled ``_mtlearn`` extension and are
usually reached through ``mtlearn.morphology`` or the primary CFP layer. They
are documented here because they appear in user-visible return values and in
advanced inspection workflows.

Morphology Trees and Filters
----------------------------

.. currentmodule:: mtlearn.morphology

.. autoclass:: WeightedMorphologicalTree
   :members:
   :show-inheritance:

.. autoclass:: Attribute
   :members:
   :show-inheritance:

.. autoclass:: AttributeFilters
   :members:
   :show-inheritance:

CFP Native Helpers
------------------

.. currentmodule:: mtlearn

.. autoclass:: ConnectedFilterPreprocessingTreeTensors
   :members:
   :show-inheritance:

.. autoclass:: ConnectedFilterPreprocessingTreeTraversal
   :members:
   :show-inheritance:

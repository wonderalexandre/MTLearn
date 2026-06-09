Morphology
==========

The ``mtlearn::morphology`` facade owns the public C++ tree, attribute, and
image-view types. Backend-specific headers remain internal implementation
details.

Installed consumers should include:

.. code-block:: cpp

   #include <mtlearn/morphology.hpp>

and link against ``mtlearn::core``.

The public header exposes morphology enums, image views, component-tree
construction helpers, and ``mtlearn::morphology::WeightedTree``. The wrapper is
the supported C++ boundary for downstream projects; direct use of backend
``mmcfilters`` headers should be treated as internal.

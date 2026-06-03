# C++ API

The public C++ API is intentionally small. Installed consumers should include:

```cpp
#include <mtlearn/morphology.hpp>
```

and link with:

```cmake
find_package(mtlearn CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE mtlearn::core)
```

## Morphology Facade

The main public namespace is `mtlearn::morphology`.

The stable tree handle is `mtlearn::morphology::WeightedTree`. It owns a
backend tree through a private implementation object and exposes construction,
basic topology queries, node altitude/residue access, pruning/merge operations,
image reconstruction, and Higra-compatible hierarchy export.

See the Doxygen group `mtlearn_morphology` for generated symbol reference.

## Backend Boundary

Code outside mtlearn should not include `mtlearn/detail/*` or backend
`mmcfilters` headers when consuming the public library. Those headers are
implementation details and are not installed by mtlearn.

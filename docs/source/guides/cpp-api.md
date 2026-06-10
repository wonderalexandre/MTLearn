# C++ API Guide

The public C++ morphology API lives in `mtlearn/morphology.hpp`. It exposes
mtlearn-owned types so downstream users do not depend on backend headers.

## Link the Core Target

Installed consumers use the exported `mtlearn::core` target.

```cmake
find_package(mtlearn REQUIRED)

add_executable(example main.cpp)
target_link_libraries(example PRIVATE mtlearn::core)
target_compile_features(example PRIVATE cxx_std_20)
```

When using mtlearn as a subdirectory, link the same target after
`add_subdirectory`.

```cmake
add_subdirectory(path/to/mtlearn)
target_link_libraries(example PRIVATE mtlearn::core)
```

## Build a Tree

`ImageViewUInt8` is a non-owning row-major image view. Keep the pixel buffer
alive until construction returns. The resulting `WeightedTree` owns the backend
tree.

```cpp
#include <cstdint>
#include <iostream>
#include <vector>

#include <mtlearn/morphology.hpp>

int main()
{
    namespace morph = mtlearn::morphology;

    std::vector<std::uint8_t> pixels = {
        0, 0, 10, 10,
        0, 30, 40, 10,
        0, 30, 40, 10,
        0, 0, 10, 10,
    };

    morph::ImageViewUInt8 image{
        pixels.data(),
        4,
        4
    };

    auto tree = morph::WeightedTree::createComponentTree(
        image,
        true  // max-tree
    );

    std::cout << tree.numRows() << "x" << tree.numCols() << "\n";
    std::cout << tree.numNodes() << " live nodes\n";
}
```

For a tree of shapes:

```cpp
auto tos = morph::WeightedTree::createTreeOfShapes(
    image,
    morph::TreeOfShapesInterpolation::SelfDual
);
```

## Query Node Values

Use `numInternalNodeSlots()` when allocating node-indexed arrays for
attributes, criteria, scores, or CFP helper data.

```cpp
std::vector<float> scores(tree.numInternalNodeSlots(), 0.0F);
```

`numNodes()` counts live topology nodes. The internal slot count can be larger,
especially after mutating operations. Direct calls to `getAltitude(node)` and
`getNodeResidue(node)` expect a valid `NodeId`; do not assume every slot is a
live node.

## Mutate and Reconstruct

The public facade exposes pruning, merging, and reconstruction. Mutating
methods require a valid `NodeId` selected by your code or by an upstream tree
inspection step.

```cpp
morph::NodeId node = 1;  // Replace with a valid node id for your tree.
tree.pruneNode(node);

morph::UInt8Image reconstructed = tree.reconstructionImage();
```

`UInt8Image::pixels` is row-major with `rows * cols` values.

## Export a Hierarchy

Use `exportHigraHierarchy()` when passing the tree to code that expects parent
and altitude vectors.

```cpp
auto [parents, altitudes] = tree.exportHigraHierarchy();
```

The two vectors have matching lengths and describe the exported hierarchy,
not necessarily the live morphology-tree node-id domain.

## API Boundaries

Downstream C++ code should include only:

```cpp
#include <mtlearn/morphology.hpp>
```

Backend headers under `detail` are intentionally not installed. If a use case
requires backend internals, add the missing operation to the public facade or
keep that code inside mtlearn.

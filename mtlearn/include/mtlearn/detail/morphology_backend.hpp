#pragma once

// Internal bridge from the public mtlearn morphology facade to the current
// backend implementation.
//
// This header is not installed as public API. It exists so implementation
// files and pybind wrappers can access the underlying mmcfilters tree without
// leaking backend types through mtlearn/morphology.hpp. If the backend is
// replaced in the future, this file should absorb most of the adaptation.

#include "mtlearn/morphology.hpp"

#include <mmcfilters/trees/MorphologicalTree.hpp>
#include <mmcfilters/trees/detail/TreeTraversalDetail.hpp>
#include <mmcfilters/trees/ValuedMorphologicalTree.hpp>
#include <mmcfilters/utils/Common.hpp>

#include <cstdint>
#include <utility>

namespace mtlearn::morphology::detail {

using TreeTopology = mmcfilters::MorphologicalTree;
using BackendWeightedTree = mmcfilters::ValuedMorphologicalTree<std::uint8_t>;

// Friend-access gateway implemented in morphology.cpp. Public code should not
// include this detail header; internal code uses it to unwrap WeightedTree only
// at the edge where backend operations are required.
struct BackendAccess {
    static BackendWeightedTree& backend(WeightedTree& tree) noexcept;
    static const BackendWeightedTree& backend(const WeightedTree& tree) noexcept;
};

// Convenience overloads used by bindings and CFP helpers. They intentionally
// return references so there is no backend copy when querying topology or
// running attribute filters.
inline BackendWeightedTree& backend(WeightedTree& tree) noexcept
{
    return BackendAccess::backend(tree);
}

inline const BackendWeightedTree& backend(const WeightedTree& tree) noexcept
{
    return BackendAccess::backend(tree);
}

inline const TreeTopology& topology(const WeightedTree& tree) noexcept
{
    return backend(tree).topology();
}

// Keep traversal calls routed through this facade helper instead of calling
// mmcfilters directly from every implementation file. This preserves one
// future migration point if the post-order traversal implementation moves.
template <class TreeLike, class PreProcessing, class MergeProcessing, class PostProcessing>
inline void traversePostOrder(
    TreeLike& tree,
    NodeId rootNodeId,
    PreProcessing&& preProcessing,
    MergeProcessing&& mergeProcessing,
    PostProcessing&& postProcessing)
{
    mmcfilters::detail::traversePostOrder(
        tree,
        rootNodeId,
        std::forward<PreProcessing>(preProcessing),
        std::forward<MergeProcessing>(mergeProcessing),
        std::forward<PostProcessing>(postProcessing));
}

} // namespace mtlearn::morphology::detail

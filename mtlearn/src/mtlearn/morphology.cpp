#include "mtlearn/morphology.hpp"

// Implementation of the public C++ morphology facade.
//
// All backend-specific construction and conversion code is kept here so the
// installed header can remain independent from mmcfilters. The Python bindings
// and CFP helpers reach the backend through detail::BackendAccess, but regular
// C++ consumers should interact with the WeightedTree facade only.

#include "mtlearn/detail/morphology_backend.hpp"

#include <mmcfilters/trees/MorphologicalTreeFactory.hpp>

#include <cstddef>
#include <stdexcept>

namespace mtlearn::morphology {
namespace {

// Convert the public non-owning image view into the backend image view. The
// backend API expects a mutable pointer even though tree construction does not
// conceptually mutate the caller's image; the cast is contained at this narrow
// boundary so the public facade can expose a const-correct view.
mmcfilters::ImageUInt8Ptr backendImageFromView(ImageViewUInt8 image)
{
    if (image.data == nullptr) {
        throw std::invalid_argument("ImageViewUInt8 data must not be null");
    }
    if (image.rows <= 0 || image.cols <= 0) {
        throw std::invalid_argument("ImageViewUInt8 dimensions must be positive");
    }

    return mmcfilters::ImageUInt8::fromExternal(
        const_cast<std::uint8_t*>(image.data),
        image.rows,
        image.cols);
}

// Translate the facade interpolation enum into the backend enum. Keeping this
// conversion local avoids exposing mmcfilters enum values in the public header.
mmcfilters::ToSInterpolation toBackendInterpolation(TreeOfShapesInterpolation interpolation)
{
    switch (interpolation) {
    case TreeOfShapesInterpolation::SelfDual:
        return mmcfilters::ToSInterpolation::SelfDual;
    case TreeOfShapesInterpolation::Min4cMax8c:
        return mmcfilters::ToSInterpolation::Min4cMax8c;
    case TreeOfShapesInterpolation::Min8cMax4c:
        return mmcfilters::ToSInterpolation::Min8cMax4c;
    }

    throw std::invalid_argument("unknown tree-of-shapes interpolation");
}

} // namespace

// Private implementation object for the shared-handle facade. The backend tree
// remains movable/copyable according to its own semantics while public
// WeightedTree instances stay cheap to copy across C++ and pybind boundaries.
struct WeightedTree::Impl {
    explicit Impl(detail::BackendWeightedTree backend) : backend(std::move(backend)) {}

    detail::BackendWeightedTree backend;
};

WeightedTree::WeightedTree(std::shared_ptr<Impl> impl) : impl_(std::move(impl))
{
    if (!impl_) {
        throw std::invalid_argument("mtlearn::morphology::WeightedTree requires a backend tree");
    }
}

WeightedTree::~WeightedTree() = default;

// Component-tree and tree-of-shapes construction are the only public creation
// points. They validate the image at the facade boundary, then delegate the
// actual morphology algorithm to the current backend.
WeightedTree WeightedTree::createComponentTree(ImageViewUInt8 image, bool isMaxTree, double radius)
{
    return WeightedTree(std::make_shared<Impl>(
        isMaxTree
            ? mmcfilters::MorphologicalTreeFactory::createMaxTree(backendImageFromView(image), radius)
            : mmcfilters::MorphologicalTreeFactory::createMinTree(backendImageFromView(image), radius)));
}

WeightedTree WeightedTree::createTreeOfShapes(
    ImageViewUInt8 image,
    TreeOfShapesInterpolation interpolation,
    int infinitySeedRow,
    int infinitySeedCol)
{
    return WeightedTree(std::make_shared<Impl>(
        mmcfilters::MorphologicalTreeFactory::createTreeOfShapes(
            backendImageFromView(image),
            toBackendInterpolation(interpolation),
            infinitySeedRow,
            infinitySeedCol)));
}

int WeightedTree::numRows() const
{
    return impl_->backend.topology().getNumRowsOfImage();
}

int WeightedTree::numCols() const
{
    return impl_->backend.topology().getNumColsOfImage();
}

int WeightedTree::numNodes() const
{
    return impl_->backend.topology().getNumNodes();
}

int WeightedTree::numInternalNodeSlots() const
{
    return impl_->backend.topology().getNumInternalNodeSlots();
}

// The following accessors intentionally forward directly to the backend. They
// keep the public API small while preserving the exact backend node-id behavior
// expected by the Python notebooks and CFP preprocessing code.
float WeightedTree::getAltitude(NodeId nodeId) const
{
    return static_cast<float>(impl_->backend.getAltitude(nodeId));
}

float WeightedTree::getNodeResidue(NodeId nodeId) const
{
    return static_cast<float>(impl_->backend.getNodeResidue(nodeId));
}

void WeightedTree::pruneNode(NodeId nodeId)
{
    impl_->backend.pruneNode(nodeId);
}

void WeightedTree::mergeNodeIntoParent(NodeId nodeId)
{
    impl_->backend.mergeNodeIntoParent(nodeId);
}

UInt8Image WeightedTree::reconstructionImage() const
{
    auto backendImage = impl_->backend.reconstructionImage();
    UInt8Image image;
    image.rows = backendImage->getNumRows();
    image.cols = backendImage->getNumCols();

    const auto buffer = backendImage->rawDataPtr();
    const auto size = static_cast<std::size_t>(image.rows) * static_cast<std::size_t>(image.cols);
    image.pixels.assign(buffer.get(), buffer.get() + size);
    return image;
}

// Convert backend export vectors into mtlearn-owned containers. This avoids
// leaking backend vector aliases or scalar types through the public API.
std::pair<std::vector<NodeId>, std::vector<float>> WeightedTree::exportHigraHierarchy() const
{
    auto [backendParent, backendAltitude] = impl_->backend.exportHigraHierarchy();
    std::vector<NodeId> parent(backendParent.begin(), backendParent.end());
    std::vector<float> altitude(backendAltitude.begin(), backendAltitude.end());
    return {std::move(parent), std::move(altitude)};
}

// BackendAccess is the single authorized unwrap point for internal code. Any
// future backend replacement should keep this section small and localized.
detail::BackendWeightedTree& detail::BackendAccess::backend(WeightedTree& tree) noexcept
{
    return tree.impl_->backend;
}

const detail::BackendWeightedTree& detail::BackendAccess::backend(const WeightedTree& tree) noexcept
{
    return tree.impl_->backend;
}

} // namespace mtlearn::morphology

#pragma once

// Shared support code for morphology pybind wrappers.
//
// This header centralizes lifetime management, NumPy conversion, validation,
// and facade-to-backend enum translation. Keeping these pieces in one file
// reduces the chance that individual bindings accidentally expose mmcfilters
// details or return arrays backed by expired C++ storage.

#include "mtlearn/detail/morphology_backend.hpp"
#include "mtlearn/morphology.hpp"

#include <mmcfilters/attributes/AttributeNames.hpp>
#include <mmcfilters/utils/Image.hpp>

#include <cstddef>
#include <cstdint>
#include <concepts>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace mtlearn {

namespace py = pybind11;
using namespace pybind11::literals;

namespace morphology_pybind {

using UInt8InputArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

enum class FloatingDType {
    Float32,
    Float64,
};

// Normalize Python dtype-like objects through NumPy. `None` preserves the
// historical mtlearn default: float32 attribute buffers.
inline FloatingDType parseFloatingDType(py::object dtype, std::string_view argumentName = "dtype")
{
    if (dtype.is_none()) {
        return FloatingDType::Float32;
    }

    py::object numpy = py::module_::import("numpy");
    py::object normalized = numpy.attr("dtype")(std::move(dtype));
    const std::string name = py::str(normalized.attr("name")).cast<std::string>();
    if (name == "float32") {
        return FloatingDType::Float32;
    }
    if (name == "float64") {
        return FloatingDType::Float64;
    }
    throw std::invalid_argument(std::string(argumentName) + " must be np.float32 or np.float64");
}

inline FloatingDType parseFloatingArrayDType(const py::array& array, std::string_view argumentName)
{
    py::object numpy = py::module_::import("numpy");
    py::object normalized = numpy.attr("dtype")(array.dtype());
    const std::string name = py::str(normalized.attr("name")).cast<std::string>();
    if (name == "float32") {
        return FloatingDType::Float32;
    }
    if (name == "float64") {
        return FloatingDType::Float64;
    }
    throw std::invalid_argument(std::string(argumentName) + " must be a 1D np.float32 or np.float64 array");
}

// Wrap a backend-owned image buffer as a NumPy array. The capsule owns a
// shared_ptr copy so the backend image memory remains alive for Python even
// after the C++ image handle leaves scope.
template <typename PixelType>
py::array_t<PixelType> imageToNumpy(mmcfilters::ImagePtr<PixelType> image)
{
    const int numCols = image->getNumColumns();
    const int numRows = image->getNumRows();
    std::shared_ptr<PixelType[]> buffer = image->rawDataPtr();

    py::capsule freeWhenDone(new std::shared_ptr<PixelType[]>(buffer), [](void* ptr) {
        delete reinterpret_cast<std::shared_ptr<PixelType[]>*>(ptr);
    });

    const py::ssize_t itemSize = sizeof(PixelType);
    return py::array_t<PixelType>(
        {static_cast<py::ssize_t>(numRows), static_cast<py::ssize_t>(numCols)},
        {static_cast<py::ssize_t>(numCols) * itemSize, itemSize},
        buffer.get(),
        freeWhenDone);
}

// Wrap an mtlearn-owned reconstruction image as a NumPy array. The pixel
// vector is moved to heap storage and released by the pybind capsule.
inline py::array_t<uint8_t> imageToNumpy(morphology::UInt8Image image)
{
    const int numRows = image.rows;
    const int numCols = image.cols;
    auto* owned = new std::vector<uint8_t>(std::move(image.pixels));

    py::capsule freeWhenDone(owned, [](void* ptr) {
        delete reinterpret_cast<std::vector<uint8_t>*>(ptr);
    });

    const py::ssize_t itemSize = sizeof(uint8_t);
    return py::array_t<uint8_t>(
        {static_cast<py::ssize_t>(numRows), static_cast<py::ssize_t>(numCols)},
        {static_cast<py::ssize_t>(numCols) * itemSize, itemSize},
        owned->data(),
        freeWhenDone);
}

// Move an owned std::vector<Real> into a NumPy array without copying. The
// capsule owns the vector and therefore controls the array's backing storage.
template <std::floating_point Real>
py::array_t<Real> vectorToNumpyOwned(std::vector<Real>&& buffer, int rows, int cols)
{
    auto* owned = new std::vector<Real>(std::move(buffer));
    py::capsule freeWhenDone(owned, [](void* ptr) {
        delete reinterpret_cast<std::vector<Real>*>(ptr);
    });

    return py::array_t<Real>(
        {rows, cols},
        {static_cast<py::ssize_t>(sizeof(Real) * cols), static_cast<py::ssize_t>(sizeof(Real))},
        owned->data(),
        freeWhenDone);
}

// One-dimensional overload used by single-attribute computations.
template <std::floating_point Real>
py::array_t<Real> vectorToNumpyOwned(std::vector<Real>&& buffer, int size)
{
    auto* owned = new std::vector<Real>(std::move(buffer));
    py::capsule freeWhenDone(owned, [](void* ptr) {
        delete reinterpret_cast<std::vector<Real>*>(ptr);
    });

    return py::array_t<Real>(
        {size},
        {static_cast<py::ssize_t>(sizeof(Real))},
        owned->data(),
        freeWhenDone);
}

// Convert Python input into the public non-owning image view expected by the
// morphology facade. py::array::c_style guarantees row-major contiguous layout.
inline morphology::ImageViewUInt8 imageViewFromArray(const UInt8InputArray& input)
{
    auto buffer = input.request();
    if (buffer.ndim != 2) {
        throw std::invalid_argument("input must be a 2D uint8 array");
    }

    return morphology::ImageViewUInt8{
        static_cast<const uint8_t*>(buffer.ptr),
        static_cast<int>(buffer.shape[0]),
        static_cast<int>(buffer.shape[1])};
}

template <std::floating_point Real>
std::shared_ptr<Real[]> floatingArrayView(const py::array_t<Real, py::array::c_style>& input)
{
    return std::shared_ptr<Real[]>(
        static_cast<Real*>(input.request().ptr),
        [owner = py::object(input)](Real*) mutable {});
}

// Validation helpers keep binding errors consistent and fail before backend
// calls receive incorrectly shaped Python data.
inline void require1DArray(const py::buffer_info& buffer, py::ssize_t expectedSize, std::string_view argumentName)
{
    if (buffer.ndim != 1) {
        std::ostringstream message;
        message << argumentName << " must be a 1D array";
        throw std::invalid_argument(message.str());
    }
    if (buffer.shape[0] != expectedSize) {
        std::ostringstream message;
        message << argumentName << " must have length " << expectedSize
                << ", got " << buffer.shape[0];
        throw std::invalid_argument(message.str());
    }
}

template <std::floating_point Real>
py::array_t<Real, py::array::c_style> require1DFloatingArray(
    py::array input,
    py::ssize_t expectedSize,
    std::string_view argumentName)
{
    const FloatingDType actualDType = parseFloatingArrayDType(input, argumentName);
    const FloatingDType expectedDType = std::same_as<Real, double>
        ? FloatingDType::Float64
        : FloatingDType::Float32;
    if (actualDType != expectedDType) {
        throw std::invalid_argument(std::string(argumentName) + " must be a 1D np.float32 or np.float64 array");
    }
    const py::buffer_info buffer = input.request();
    require1DArray(buffer, expectedSize, argumentName);
    if (buffer.strides[0] != static_cast<py::ssize_t>(sizeof(Real))) {
        throw std::invalid_argument(std::string(argumentName) + " must be C-contiguous");
    }
    return py::reinterpret_borrow<py::array_t<Real, py::array::c_style>>(input);
}

template <class T>
void requireVectorSize(const std::vector<T>& values, std::size_t expectedSize, std::string_view argumentName)
{
    if (values.size() != expectedSize) {
        std::ostringstream message;
        message << argumentName << " must have length " << expectedSize
                << ", got " << values.size();
        throw std::invalid_argument(message.str());
    }
}

// The following conversion functions are the authoritative mapping from the
// mtlearn public C++ facade to mmcfilters. Whenever a public enum is extended,
// update this table and the corresponding Python enum exposure together.
inline mmcfilters::NodeIdSpace toBackend(morphology::NodeIdSpace outputSpace)
{
    switch (outputSpace) {
    case morphology::NodeIdSpace::MORPHOLOGICAL_TREE:
        return mmcfilters::NodeIdSpace::MorphologicalTree;
    case morphology::NodeIdSpace::HIGRA:
        return mmcfilters::NodeIdSpace::Higra;
    }
    throw std::invalid_argument("unknown NodeIdSpace");
}

inline mmcfilters::AttributeGroup toBackend(morphology::AttributeGroup group)
{
    switch (group) {
    case morphology::AttributeGroup::ALL:
        return mmcfilters::AttributeGroup::All;
    case morphology::AttributeGroup::GRAY_LEVEL:
        return mmcfilters::AttributeGroup::GrayLevel;
    case morphology::AttributeGroup::SHAPE:
        return mmcfilters::AttributeGroup::Shape;
    case morphology::AttributeGroup::MOMENTS:
        return mmcfilters::AttributeGroup::Moments;
    case morphology::AttributeGroup::BOUNDARY:
        return mmcfilters::AttributeGroup::Boundary;
    case morphology::AttributeGroup::TREE_TOPOLOGY:
        return mmcfilters::AttributeGroup::TreeTopology;
    }
    throw std::invalid_argument("unknown AttributeGroup");
}

inline mmcfilters::Attribute toBackend(morphology::Attribute attribute)
{
    switch (attribute) {
    case morphology::Attribute::AREA:
        return mmcfilters::Attribute::Area;
    case morphology::Attribute::VOLUME:
        return mmcfilters::Attribute::Volume;
    case morphology::Attribute::RELATIVE_VOLUME:
        return mmcfilters::Attribute::RelativeVolume;
    // PENDING(decisao): Attribute::LEVEL nao tem correspondente no enum novo.
    case morphology::Attribute::GRAY_HEIGHT:
        return mmcfilters::Attribute::GrayLevelHeight;
    case morphology::Attribute::MEAN_LEVEL:
        return mmcfilters::Attribute::MeanGrayLevel;
    case morphology::Attribute::VARIANCE_LEVEL:
        return mmcfilters::Attribute::GrayLevelVariance;
    case morphology::Attribute::BOX_WIDTH:
        return mmcfilters::Attribute::BoxWidth;
    case morphology::Attribute::BOX_HEIGHT:
        return mmcfilters::Attribute::BoundingBoxHeight;
    case morphology::Attribute::DIAGONAL_LENGTH:
        return mmcfilters::Attribute::DiagonalLength;
    case morphology::Attribute::RECTANGULARITY:
        return mmcfilters::Attribute::Rectangularity;
    case morphology::Attribute::RATIO_WH:
        return mmcfilters::Attribute::RatioWh;
    case morphology::Attribute::BOX_COL_MIN:
        return mmcfilters::Attribute::BoxColumnMin;
    case morphology::Attribute::BOX_COL_MAX:
        return mmcfilters::Attribute::BoxColumnMax;
    case morphology::Attribute::BOX_ROW_MIN:
        return mmcfilters::Attribute::BoxRowMin;
    case morphology::Attribute::BOX_ROW_MAX:
        return mmcfilters::Attribute::BoxRowMax;
    case morphology::Attribute::CENTRAL_MOMENT_20:
        return mmcfilters::Attribute::CentralMoment20;
    case morphology::Attribute::CENTRAL_MOMENT_02:
        return mmcfilters::Attribute::CentralMoment02;
    case morphology::Attribute::CENTRAL_MOMENT_11:
        return mmcfilters::Attribute::CentralMoment11;
    case morphology::Attribute::CENTRAL_MOMENT_30:
        return mmcfilters::Attribute::CentralMoment30;
    case morphology::Attribute::CENTRAL_MOMENT_03:
        return mmcfilters::Attribute::CentralMoment03;
    case morphology::Attribute::CENTRAL_MOMENT_21:
        return mmcfilters::Attribute::CentralMoment21;
    case morphology::Attribute::CENTRAL_MOMENT_12:
        return mmcfilters::Attribute::CentralMoment12;
    case morphology::Attribute::HU_MOMENT_1:
        return mmcfilters::Attribute::HuMoment1;
    case morphology::Attribute::HU_MOMENT_2:
        return mmcfilters::Attribute::HuMoment2;
    case morphology::Attribute::HU_MOMENT_3:
        return mmcfilters::Attribute::HuMoment3;
    case morphology::Attribute::HU_MOMENT_4:
        return mmcfilters::Attribute::HuMoment4;
    case morphology::Attribute::HU_MOMENT_5:
        return mmcfilters::Attribute::HuMoment5;
    case morphology::Attribute::HU_MOMENT_6:
        return mmcfilters::Attribute::HuMoment6;
    case morphology::Attribute::HU_MOMENT_7:
        return mmcfilters::Attribute::HuMoment7;
    case morphology::Attribute::INERTIA:
        return mmcfilters::Attribute::Inertia;
    case morphology::Attribute::COMPACTNESS:
        return mmcfilters::Attribute::Compactness;
    case morphology::Attribute::ECCENTRICITY:
        return mmcfilters::Attribute::Eccentricity;
    case morphology::Attribute::LENGTH_MAJOR_AXIS:
        return mmcfilters::Attribute::LengthMajorAxis;
    case morphology::Attribute::LENGTH_MINOR_AXIS:
        return mmcfilters::Attribute::LengthMinorAxis;
    case morphology::Attribute::AXIS_ORIENTATION:
        return mmcfilters::Attribute::AxisOrientation;
    case morphology::Attribute::CIRCULARITY:
        return mmcfilters::Attribute::Circularity;
    case morphology::Attribute::BITQUADS_AREA:
        return mmcfilters::Attribute::BitquadArea;
    case morphology::Attribute::BITQUADS_NUMBER_EULER:
        return mmcfilters::Attribute::BitquadNumberEuler;
    case morphology::Attribute::BITQUADS_NUMBER_HOLES:
        return mmcfilters::Attribute::BitquadNumberHoles;
    case morphology::Attribute::BITQUADS_PERIMETER:
        return mmcfilters::Attribute::BitquadPerimeter;
    case morphology::Attribute::BITQUADS_PERIMETER_CONTINUOUS:
        return mmcfilters::Attribute::BitquadPerimeterContinuous;
    case morphology::Attribute::BITQUADS_CIRCULARITY:
        return mmcfilters::Attribute::BitquadCircularity;
    case morphology::Attribute::BITQUADS_PERIMETER_AVERAGE:
        return mmcfilters::Attribute::BitquadPerimeterAverage;
    case morphology::Attribute::BITQUADS_LENGTH_AVERAGE:
        return mmcfilters::Attribute::BitquadLengthAverage;
    case morphology::Attribute::BITQUADS_WIDTH_AVERAGE:
        return mmcfilters::Attribute::BitquadWidthAverage;
    case morphology::Attribute::HEIGHT_NODE:
        return mmcfilters::Attribute::SubtreeHeight;
    case morphology::Attribute::DEPTH_NODE:
        return mmcfilters::Attribute::DepthNode;
    case morphology::Attribute::IS_LEAF_NODE:
        return mmcfilters::Attribute::IsLeafNode;
    case morphology::Attribute::IS_ROOT_NODE:
        return mmcfilters::Attribute::IsRootNode;
    case morphology::Attribute::NUM_CHILDREN_NODE:
        return mmcfilters::Attribute::NumChildrenNode;
    case morphology::Attribute::NUM_SIBLINGS_NODE:
        return mmcfilters::Attribute::NumSiblingsNode;
    case morphology::Attribute::NUM_DESCENDANTS_NODE:
        return mmcfilters::Attribute::NumDescendantsNode;
    case morphology::Attribute::NUM_LEAF_DESCENDANTS_NODE:
        return mmcfilters::Attribute::NumLeafDescendantsNode;
    case morphology::Attribute::LEAF_RATIO_NODE:
        return mmcfilters::Attribute::LeafRatioNode;
    case morphology::Attribute::BALANCE_NODE:
        return mmcfilters::Attribute::BalanceNode;
    case morphology::Attribute::MAX_DIST:
        return mmcfilters::Attribute::MaxDist;
    case morphology::Attribute::AVG_CHILD_HEIGHT_NODE:
        return mmcfilters::Attribute::AvgChildHeightNode;
    case morphology::Attribute::CONTOUR_PIXELS:
        return mmcfilters::Attribute::ContourPixels;
    case morphology::Attribute::CONTOUR_PERIMETER:
        return mmcfilters::Attribute::ContourPerimeter;
    case morphology::Attribute::CONTOUR_SIDE_NORTH:
        return mmcfilters::Attribute::ContourSideNorth;
    case morphology::Attribute::CONTOUR_SIDE_WEST:
        return mmcfilters::Attribute::ContourSideWest;
    case morphology::Attribute::CONTOUR_SIDE_EAST:
        return mmcfilters::Attribute::ContourSideEast;
    case morphology::Attribute::CONTOUR_SIDE_SOUTH:
        return mmcfilters::Attribute::ContourSideSouth;
    }
    throw std::invalid_argument("unknown Attribute");
}

inline morphology::Attribute fromBackend(mmcfilters::Attribute attribute)
{
// The facade keeps its own SCREAMING_CASE vocabulary, so the backend and facade
// enumerators no longer share a spelling and must both be named explicitly.
#define MTLEARN_FROM_BACKEND_ATTRIBUTE(backendName, facadeName) \
    case mmcfilters::Attribute::backendName:                    \
        return morphology::Attribute::facadeName

    switch (attribute) {
    MTLEARN_FROM_BACKEND_ATTRIBUTE(Area, AREA);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(Volume, VOLUME);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(RelativeVolume, RELATIVE_VOLUME);
    // PENDING(decisao): Attribute::LEVEL nao tem correspondente no enum novo.
    MTLEARN_FROM_BACKEND_ATTRIBUTE(GrayLevelHeight, GRAY_HEIGHT);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MeanGrayLevel, MEAN_LEVEL);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(GrayLevelVariance, VARIANCE_LEVEL);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BoxWidth, BOX_WIDTH);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BoundingBoxHeight, BOX_HEIGHT);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DiagonalLength, DIAGONAL_LENGTH);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(Rectangularity, RECTANGULARITY);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(RatioWh, RATIO_WH);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BoxColumnMin, BOX_COL_MIN);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BoxColumnMax, BOX_COL_MAX);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BoxRowMin, BOX_ROW_MIN);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BoxRowMax, BOX_ROW_MAX);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(CentralMoment20, CENTRAL_MOMENT_20);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(CentralMoment02, CENTRAL_MOMENT_02);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(CentralMoment11, CENTRAL_MOMENT_11);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(CentralMoment30, CENTRAL_MOMENT_30);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(CentralMoment03, CENTRAL_MOMENT_03);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(CentralMoment21, CENTRAL_MOMENT_21);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(CentralMoment12, CENTRAL_MOMENT_12);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(HuMoment1, HU_MOMENT_1);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(HuMoment2, HU_MOMENT_2);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(HuMoment3, HU_MOMENT_3);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(HuMoment4, HU_MOMENT_4);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(HuMoment5, HU_MOMENT_5);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(HuMoment6, HU_MOMENT_6);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(HuMoment7, HU_MOMENT_7);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(Inertia, INERTIA);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(Compactness, COMPACTNESS);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(Eccentricity, ECCENTRICITY);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(LengthMajorAxis, LENGTH_MAJOR_AXIS);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(LengthMinorAxis, LENGTH_MINOR_AXIS);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(AxisOrientation, AXIS_ORIENTATION);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(Circularity, CIRCULARITY);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadArea, BITQUADS_AREA);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadNumberEuler, BITQUADS_NUMBER_EULER);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadNumberHoles, BITQUADS_NUMBER_HOLES);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadPerimeter, BITQUADS_PERIMETER);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadPerimeterContinuous, BITQUADS_PERIMETER_CONTINUOUS);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadCircularity, BITQUADS_CIRCULARITY);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadPerimeterAverage, BITQUADS_PERIMETER_AVERAGE);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadLengthAverage, BITQUADS_LENGTH_AVERAGE);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadWidthAverage, BITQUADS_WIDTH_AVERAGE);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(SubtreeHeight, HEIGHT_NODE);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DepthNode, DEPTH_NODE);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(IsLeafNode, IS_LEAF_NODE);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(IsRootNode, IS_ROOT_NODE);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(NumChildrenNode, NUM_CHILDREN_NODE);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(NumSiblingsNode, NUM_SIBLINGS_NODE);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(NumDescendantsNode, NUM_DESCENDANTS_NODE);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(NumLeafDescendantsNode, NUM_LEAF_DESCENDANTS_NODE);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(LeafRatioNode, LEAF_RATIO_NODE);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BalanceNode, BALANCE_NODE);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MaxDist, MAX_DIST);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(AvgChildHeightNode, AVG_CHILD_HEIGHT_NODE);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(ContourPixels, CONTOUR_PIXELS);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(ContourPerimeter, CONTOUR_PERIMETER);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(ContourSideNorth, CONTOUR_SIDE_NORTH);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(ContourSideWest, CONTOUR_SIDE_WEST);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(ContourSideEast, CONTOUR_SIDE_EAST);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(ContourSideSouth, CONTOUR_SIDE_SOUTH);
    }

#undef MTLEARN_FROM_BACKEND_ATTRIBUTE

    throw std::invalid_argument("unknown backend Attribute");
}

inline mmcfilters::AttributeOrGroup toBackend(morphology::AttributeOrGroup attribute)
{
    // Variants allow Python/C++ callers to request either one concrete
    // attribute or a backend-expanded attribute group through a single API.
    return std::visit(
        [](auto value) -> mmcfilters::AttributeOrGroup {
            return toBackend(value);
        },
        attribute);
}

inline std::vector<mmcfilters::AttributeOrGroup> toBackend(const std::vector<morphology::AttributeOrGroup>& attributes)
{
    // Preserve request order; the backend AttributeNames object determines the
    // final output column mapping returned to Python.
    std::vector<mmcfilters::AttributeOrGroup> result;
    result.reserve(attributes.size());
    for (const auto& attribute : attributes) {
        result.push_back(toBackend(attribute));
    }
    return result;
}

} // namespace morphology_pybind
} // namespace mtlearn

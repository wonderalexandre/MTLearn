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
    case morphology::NodeIdSpace::MorphologicalTree:
        return mmcfilters::NodeIdSpace::MorphologicalTree;
    case morphology::NodeIdSpace::Higra:
        return mmcfilters::NodeIdSpace::Higra;
    }
    throw std::invalid_argument("unknown NodeIdSpace");
}

inline mmcfilters::AttributeGroup toBackend(morphology::AttributeGroup group)
{
    switch (group) {
    case morphology::AttributeGroup::All:
        return mmcfilters::AttributeGroup::All;
    case morphology::AttributeGroup::GrayLevel:
        return mmcfilters::AttributeGroup::GrayLevel;
    case morphology::AttributeGroup::Shape:
        return mmcfilters::AttributeGroup::Shape;
    case morphology::AttributeGroup::Moments:
        return mmcfilters::AttributeGroup::Moments;
    case morphology::AttributeGroup::Boundary:
        return mmcfilters::AttributeGroup::Boundary;
    case morphology::AttributeGroup::TreeTopology:
        return mmcfilters::AttributeGroup::TreeTopology;
    case morphology::AttributeGroup::DistTransf:
        return mmcfilters::AttributeGroup::DistTransf;
    case morphology::AttributeGroup::DistTransfExact:
        return mmcfilters::AttributeGroup::DistTransfExact;
    case morphology::AttributeGroup::FilledShape:
        return mmcfilters::AttributeGroup::FilledShape;
    }
    throw std::invalid_argument("unknown AttributeGroup");
}

inline mmcfilters::Attribute toBackend(morphology::Attribute attribute)
{
    switch (attribute) {
    case morphology::Attribute::Area:
        return mmcfilters::Attribute::Area;
    case morphology::Attribute::Volume:
        return mmcfilters::Attribute::Volume;
    case morphology::Attribute::RelativeVolume:
        return mmcfilters::Attribute::RelativeVolume;
    // The backend dropped the plain node-altitude attribute, so the facade no
    // longer publishes LEVEL/ALTITUDE either.
    case morphology::Attribute::GrayLevelHeight:
        return mmcfilters::Attribute::GrayLevelHeight;
    case morphology::Attribute::MeanGrayLevel:
        return mmcfilters::Attribute::MeanGrayLevel;
    case morphology::Attribute::GrayLevelVariance:
        return mmcfilters::Attribute::GrayLevelVariance;
    case morphology::Attribute::BoxWidth:
        return mmcfilters::Attribute::BoxWidth;
    case morphology::Attribute::BoundingBoxHeight:
        return mmcfilters::Attribute::BoundingBoxHeight;
    case morphology::Attribute::DiagonalLength:
        return mmcfilters::Attribute::DiagonalLength;
    case morphology::Attribute::Rectangularity:
        return mmcfilters::Attribute::Rectangularity;
    case morphology::Attribute::RatioWh:
        return mmcfilters::Attribute::RatioWh;
    case morphology::Attribute::BoxColumnMin:
        return mmcfilters::Attribute::BoxColumnMin;
    case morphology::Attribute::BoxColumnMax:
        return mmcfilters::Attribute::BoxColumnMax;
    case morphology::Attribute::BoxRowMin:
        return mmcfilters::Attribute::BoxRowMin;
    case morphology::Attribute::BoxRowMax:
        return mmcfilters::Attribute::BoxRowMax;
    case morphology::Attribute::CentralMoment20:
        return mmcfilters::Attribute::CentralMoment20;
    case morphology::Attribute::CentralMoment02:
        return mmcfilters::Attribute::CentralMoment02;
    case morphology::Attribute::CentralMoment11:
        return mmcfilters::Attribute::CentralMoment11;
    case morphology::Attribute::CentralMoment30:
        return mmcfilters::Attribute::CentralMoment30;
    case morphology::Attribute::CentralMoment03:
        return mmcfilters::Attribute::CentralMoment03;
    case morphology::Attribute::CentralMoment21:
        return mmcfilters::Attribute::CentralMoment21;
    case morphology::Attribute::CentralMoment12:
        return mmcfilters::Attribute::CentralMoment12;
    case morphology::Attribute::HuMoment1:
        return mmcfilters::Attribute::HuMoment1;
    case morphology::Attribute::HuMoment2:
        return mmcfilters::Attribute::HuMoment2;
    case morphology::Attribute::HuMoment3:
        return mmcfilters::Attribute::HuMoment3;
    case morphology::Attribute::HuMoment4:
        return mmcfilters::Attribute::HuMoment4;
    case morphology::Attribute::HuMoment5:
        return mmcfilters::Attribute::HuMoment5;
    case morphology::Attribute::HuMoment6:
        return mmcfilters::Attribute::HuMoment6;
    case morphology::Attribute::HuMoment7:
        return mmcfilters::Attribute::HuMoment7;
    case morphology::Attribute::Inertia:
        return mmcfilters::Attribute::Inertia;
    case morphology::Attribute::Compactness:
        return mmcfilters::Attribute::Compactness;
    case morphology::Attribute::Eccentricity:
        return mmcfilters::Attribute::Eccentricity;
    case morphology::Attribute::LengthMajorAxis:
        return mmcfilters::Attribute::LengthMajorAxis;
    case morphology::Attribute::LengthMinorAxis:
        return mmcfilters::Attribute::LengthMinorAxis;
    case morphology::Attribute::AxisOrientation:
        return mmcfilters::Attribute::AxisOrientation;
    case morphology::Attribute::Circularity:
        return mmcfilters::Attribute::Circularity;
    case morphology::Attribute::BitquadArea:
        return mmcfilters::Attribute::BitquadArea;
    case morphology::Attribute::BitquadNumberEuler:
        return mmcfilters::Attribute::BitquadNumberEuler;
    case morphology::Attribute::BitquadNumberHoles:
        return mmcfilters::Attribute::BitquadNumberHoles;
    case morphology::Attribute::BitquadPerimeter:
        return mmcfilters::Attribute::BitquadPerimeter;
    case morphology::Attribute::BitquadPerimeterContinuous:
        return mmcfilters::Attribute::BitquadPerimeterContinuous;
    case morphology::Attribute::BitquadCircularity:
        return mmcfilters::Attribute::BitquadCircularity;
    case morphology::Attribute::BitquadPerimeterAverage:
        return mmcfilters::Attribute::BitquadPerimeterAverage;
    case morphology::Attribute::BitquadLengthAverage:
        return mmcfilters::Attribute::BitquadLengthAverage;
    case morphology::Attribute::BitquadWidthAverage:
        return mmcfilters::Attribute::BitquadWidthAverage;
    case morphology::Attribute::SubtreeHeight:
        return mmcfilters::Attribute::SubtreeHeight;
    case morphology::Attribute::DepthNode:
        return mmcfilters::Attribute::DepthNode;
    case morphology::Attribute::IsLeafNode:
        return mmcfilters::Attribute::IsLeafNode;
    case morphology::Attribute::IsRootNode:
        return mmcfilters::Attribute::IsRootNode;
    case morphology::Attribute::NumChildrenNode:
        return mmcfilters::Attribute::NumChildrenNode;
    case morphology::Attribute::NumSiblingsNode:
        return mmcfilters::Attribute::NumSiblingsNode;
    case morphology::Attribute::NumDescendantsNode:
        return mmcfilters::Attribute::NumDescendantsNode;
    case morphology::Attribute::NumLeafDescendantsNode:
        return mmcfilters::Attribute::NumLeafDescendantsNode;
    case morphology::Attribute::LeafRatioNode:
        return mmcfilters::Attribute::LeafRatioNode;
    case morphology::Attribute::BalanceNode:
        return mmcfilters::Attribute::BalanceNode;
    case morphology::Attribute::MaxDist:
        return mmcfilters::Attribute::MaxDist;
    case morphology::Attribute::AvgChildHeightNode:
        return mmcfilters::Attribute::AvgChildHeightNode;
    case morphology::Attribute::ContourPixels:
        return mmcfilters::Attribute::ContourPixels;
    case morphology::Attribute::ContourPerimeter:
        return mmcfilters::Attribute::ContourPerimeter;
    case morphology::Attribute::ContourSideNorth:
        return mmcfilters::Attribute::ContourSideNorth;
    case morphology::Attribute::ContourSideWest:
        return mmcfilters::Attribute::ContourSideWest;
    case morphology::Attribute::ContourSideEast:
        return mmcfilters::Attribute::ContourSideEast;
    case morphology::Attribute::ContourSideSouth:
        return mmcfilters::Attribute::ContourSideSouth;
    case morphology::Attribute::MaxDistExact:
        return mmcfilters::Attribute::MaxDistExact;
    case morphology::Attribute::DistSquaredSumExact:
        return mmcfilters::Attribute::DistSquaredSumExact;
    case morphology::Attribute::DistSquaredMeanExact:
        return mmcfilters::Attribute::DistSquaredMeanExact;
    case morphology::Attribute::DistRmsExact:
        return mmcfilters::Attribute::DistRmsExact;
    case morphology::Attribute::DistSquaredVarianceExact:
        return mmcfilters::Attribute::DistSquaredVarianceExact;
    case morphology::Attribute::DistSquaredSum:
        return mmcfilters::Attribute::DistSquaredSum;
    case morphology::Attribute::DistSquaredMean:
        return mmcfilters::Attribute::DistSquaredMean;
    case morphology::Attribute::DistRms:
        return mmcfilters::Attribute::DistRms;
    case morphology::Attribute::DistSquaredVariance:
        return mmcfilters::Attribute::DistSquaredVariance;
    case morphology::Attribute::MaxDistCenterRowExact:
        return mmcfilters::Attribute::MaxDistCenterRowExact;
    case morphology::Attribute::MaxDistCenterColumnExact:
        return mmcfilters::Attribute::MaxDistCenterColumnExact;
    case morphology::Attribute::MaxDistCenterRow:
        return mmcfilters::Attribute::MaxDistCenterRow;
    case morphology::Attribute::MaxDistCenterColumn:
        return mmcfilters::Attribute::MaxDistCenterColumn;
    case morphology::Attribute::MaxDistPlateauAreaExact:
        return mmcfilters::Attribute::MaxDistPlateauAreaExact;
    case morphology::Attribute::MaxDistPlateauCentroidRowExact:
        return mmcfilters::Attribute::MaxDistPlateauCentroidRowExact;
    case morphology::Attribute::MaxDistPlateauCentroidColumnExact:
        return mmcfilters::Attribute::MaxDistPlateauCentroidColumnExact;
    case morphology::Attribute::MaxDistPlateauArea:
        return mmcfilters::Attribute::MaxDistPlateauArea;
    case morphology::Attribute::MaxDistPlateauCentroidRow:
        return mmcfilters::Attribute::MaxDistPlateauCentroidRow;
    case morphology::Attribute::MaxDistPlateauCentroidColumn:
        return mmcfilters::Attribute::MaxDistPlateauCentroidColumn;
    case morphology::Attribute::DistSum:
        return mmcfilters::Attribute::DistSum;
    case morphology::Attribute::DistMean:
        return mmcfilters::Attribute::DistMean;
    case morphology::Attribute::DistVariance:
        return mmcfilters::Attribute::DistVariance;
    case morphology::Attribute::DistMedian:
        return mmcfilters::Attribute::DistMedian;
    case morphology::Attribute::DistMode:
        return mmcfilters::Attribute::DistMode;
    case morphology::Attribute::DistQ25:
        return mmcfilters::Attribute::DistQ25;
    case morphology::Attribute::DistQ75:
        return mmcfilters::Attribute::DistQ75;
    case morphology::Attribute::DistQ90:
        return mmcfilters::Attribute::DistQ90;
    case morphology::Attribute::DistEntropy:
        return mmcfilters::Attribute::DistEntropy;
    case morphology::Attribute::DistPositiveArea:
        return mmcfilters::Attribute::DistPositiveArea;
    case morphology::Attribute::DistLevelCount:
        return mmcfilters::Attribute::DistLevelCount;
    case morphology::Attribute::DistWeightedCentroidRow:
        return mmcfilters::Attribute::DistWeightedCentroidRow;
    case morphology::Attribute::DistWeightedCentroidColumn:
        return mmcfilters::Attribute::DistWeightedCentroidColumn;
    case morphology::Attribute::DistWeightedCentralMoment20:
        return mmcfilters::Attribute::DistWeightedCentralMoment20;
    case morphology::Attribute::DistWeightedCentralMoment02:
        return mmcfilters::Attribute::DistWeightedCentralMoment02;
    case morphology::Attribute::DistWeightedCentralMoment11:
        return mmcfilters::Attribute::DistWeightedCentralMoment11;
    case morphology::Attribute::DistWeightedAxisOrientation:
        return mmcfilters::Attribute::DistWeightedAxisOrientation;
    case morphology::Attribute::DistWeightedEccentricity:
        return mmcfilters::Attribute::DistWeightedEccentricity;
    case morphology::Attribute::DistSumExact:
        return mmcfilters::Attribute::DistSumExact;
    case morphology::Attribute::DistMeanExact:
        return mmcfilters::Attribute::DistMeanExact;
    case morphology::Attribute::DistVarianceExact:
        return mmcfilters::Attribute::DistVarianceExact;
    case morphology::Attribute::DistMedianExact:
        return mmcfilters::Attribute::DistMedianExact;
    case morphology::Attribute::DistModeExact:
        return mmcfilters::Attribute::DistModeExact;
    case morphology::Attribute::DistQ25Exact:
        return mmcfilters::Attribute::DistQ25Exact;
    case morphology::Attribute::DistQ75Exact:
        return mmcfilters::Attribute::DistQ75Exact;
    case morphology::Attribute::DistQ90Exact:
        return mmcfilters::Attribute::DistQ90Exact;
    case morphology::Attribute::DistEntropyExact:
        return mmcfilters::Attribute::DistEntropyExact;
    case morphology::Attribute::DistPositiveAreaExact:
        return mmcfilters::Attribute::DistPositiveAreaExact;
    case morphology::Attribute::DistLevelCountExact:
        return mmcfilters::Attribute::DistLevelCountExact;
    case morphology::Attribute::DistWeightedCentroidRowExact:
        return mmcfilters::Attribute::DistWeightedCentroidRowExact;
    case morphology::Attribute::DistWeightedCentroidColumnExact:
        return mmcfilters::Attribute::DistWeightedCentroidColumnExact;
    case morphology::Attribute::DistWeightedCentralMoment20Exact:
        return mmcfilters::Attribute::DistWeightedCentralMoment20Exact;
    case morphology::Attribute::DistWeightedCentralMoment02Exact:
        return mmcfilters::Attribute::DistWeightedCentralMoment02Exact;
    case morphology::Attribute::DistWeightedCentralMoment11Exact:
        return mmcfilters::Attribute::DistWeightedCentralMoment11Exact;
    case morphology::Attribute::DistWeightedAxisOrientationExact:
        return mmcfilters::Attribute::DistWeightedAxisOrientationExact;
    case morphology::Attribute::DistWeightedEccentricityExact:
        return mmcfilters::Attribute::DistWeightedEccentricityExact;
    case morphology::Attribute::MaxSquaredDist:
        return mmcfilters::Attribute::MaxSquaredDist;
    case morphology::Attribute::MaxSquaredDistExact:
        return mmcfilters::Attribute::MaxSquaredDistExact;
    case morphology::Attribute::FilledArea:
        return mmcfilters::Attribute::FilledArea;
    case morphology::Attribute::FilledCentroidRow:
        return mmcfilters::Attribute::FilledCentroidRow;
    case morphology::Attribute::FilledCentroidColumn:
        return mmcfilters::Attribute::FilledCentroidColumn;
    case morphology::Attribute::FilledLengthMajorAxis:
        return mmcfilters::Attribute::FilledLengthMajorAxis;
    case morphology::Attribute::FilledLengthMinorAxis:
        return mmcfilters::Attribute::FilledLengthMinorAxis;
    case morphology::Attribute::FilledAxisOrientation:
        return mmcfilters::Attribute::FilledAxisOrientation;
    case morphology::Attribute::FilledEccentricity:
        return mmcfilters::Attribute::FilledEccentricity;
    case morphology::Attribute::FilledInertia:
        return mmcfilters::Attribute::FilledInertia;
    case morphology::Attribute::HoleAreaFraction:
        return mmcfilters::Attribute::HoleAreaFraction;
    case morphology::Attribute::FilledCentroidDisplacementNormalized:
        return mmcfilters::Attribute::FilledCentroidDisplacementNormalized;
    case morphology::Attribute::FilledCompactness:
        return mmcfilters::Attribute::FilledCompactness;
    case morphology::Attribute::FilledCircularity:
        return mmcfilters::Attribute::FilledCircularity;
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
    MTLEARN_FROM_BACKEND_ATTRIBUTE(Area, Area);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(Volume, Volume);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(RelativeVolume, RelativeVolume);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(GrayLevelHeight, GrayLevelHeight);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MeanGrayLevel, MeanGrayLevel);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(GrayLevelVariance, GrayLevelVariance);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BoxWidth, BoxWidth);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BoundingBoxHeight, BoundingBoxHeight);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DiagonalLength, DiagonalLength);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(Rectangularity, Rectangularity);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(RatioWh, RatioWh);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BoxColumnMin, BoxColumnMin);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BoxColumnMax, BoxColumnMax);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BoxRowMin, BoxRowMin);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BoxRowMax, BoxRowMax);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(CentralMoment20, CentralMoment20);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(CentralMoment02, CentralMoment02);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(CentralMoment11, CentralMoment11);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(CentralMoment30, CentralMoment30);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(CentralMoment03, CentralMoment03);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(CentralMoment21, CentralMoment21);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(CentralMoment12, CentralMoment12);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(HuMoment1, HuMoment1);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(HuMoment2, HuMoment2);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(HuMoment3, HuMoment3);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(HuMoment4, HuMoment4);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(HuMoment5, HuMoment5);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(HuMoment6, HuMoment6);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(HuMoment7, HuMoment7);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(Inertia, Inertia);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(Compactness, Compactness);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(Eccentricity, Eccentricity);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(LengthMajorAxis, LengthMajorAxis);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(LengthMinorAxis, LengthMinorAxis);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(AxisOrientation, AxisOrientation);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(Circularity, Circularity);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadArea, BitquadArea);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadNumberEuler, BitquadNumberEuler);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadNumberHoles, BitquadNumberHoles);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadPerimeter, BitquadPerimeter);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadPerimeterContinuous, BitquadPerimeterContinuous);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadCircularity, BitquadCircularity);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadPerimeterAverage, BitquadPerimeterAverage);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadLengthAverage, BitquadLengthAverage);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BitquadWidthAverage, BitquadWidthAverage);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(SubtreeHeight, SubtreeHeight);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DepthNode, DepthNode);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(IsLeafNode, IsLeafNode);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(IsRootNode, IsRootNode);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(NumChildrenNode, NumChildrenNode);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(NumSiblingsNode, NumSiblingsNode);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(NumDescendantsNode, NumDescendantsNode);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(NumLeafDescendantsNode, NumLeafDescendantsNode);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(LeafRatioNode, LeafRatioNode);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(BalanceNode, BalanceNode);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MaxDist, MaxDist);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(AvgChildHeightNode, AvgChildHeightNode);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(ContourPixels, ContourPixels);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(ContourPerimeter, ContourPerimeter);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(ContourSideNorth, ContourSideNorth);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(ContourSideWest, ContourSideWest);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(ContourSideEast, ContourSideEast);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(ContourSideSouth, ContourSideSouth);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MaxDistExact, MaxDistExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistSquaredSumExact, DistSquaredSumExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistSquaredMeanExact, DistSquaredMeanExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistRmsExact, DistRmsExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistSquaredVarianceExact, DistSquaredVarianceExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistSquaredSum, DistSquaredSum);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistSquaredMean, DistSquaredMean);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistRms, DistRms);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistSquaredVariance, DistSquaredVariance);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MaxDistCenterRowExact, MaxDistCenterRowExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MaxDistCenterColumnExact, MaxDistCenterColumnExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MaxDistCenterRow, MaxDistCenterRow);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MaxDistCenterColumn, MaxDistCenterColumn);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MaxDistPlateauAreaExact, MaxDistPlateauAreaExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MaxDistPlateauCentroidRowExact, MaxDistPlateauCentroidRowExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MaxDistPlateauCentroidColumnExact, MaxDistPlateauCentroidColumnExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MaxDistPlateauArea, MaxDistPlateauArea);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MaxDistPlateauCentroidRow, MaxDistPlateauCentroidRow);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MaxDistPlateauCentroidColumn, MaxDistPlateauCentroidColumn);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistSum, DistSum);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistMean, DistMean);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistVariance, DistVariance);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistMedian, DistMedian);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistMode, DistMode);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistQ25, DistQ25);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistQ75, DistQ75);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistQ90, DistQ90);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistEntropy, DistEntropy);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistPositiveArea, DistPositiveArea);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistLevelCount, DistLevelCount);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistWeightedCentroidRow, DistWeightedCentroidRow);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistWeightedCentroidColumn, DistWeightedCentroidColumn);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistWeightedCentralMoment20, DistWeightedCentralMoment20);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistWeightedCentralMoment02, DistWeightedCentralMoment02);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistWeightedCentralMoment11, DistWeightedCentralMoment11);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistWeightedAxisOrientation, DistWeightedAxisOrientation);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistWeightedEccentricity, DistWeightedEccentricity);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistSumExact, DistSumExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistMeanExact, DistMeanExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistVarianceExact, DistVarianceExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistMedianExact, DistMedianExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistModeExact, DistModeExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistQ25Exact, DistQ25Exact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistQ75Exact, DistQ75Exact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistQ90Exact, DistQ90Exact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistEntropyExact, DistEntropyExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistPositiveAreaExact, DistPositiveAreaExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistLevelCountExact, DistLevelCountExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistWeightedCentroidRowExact, DistWeightedCentroidRowExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistWeightedCentroidColumnExact, DistWeightedCentroidColumnExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistWeightedCentralMoment20Exact, DistWeightedCentralMoment20Exact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistWeightedCentralMoment02Exact, DistWeightedCentralMoment02Exact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistWeightedCentralMoment11Exact, DistWeightedCentralMoment11Exact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistWeightedAxisOrientationExact, DistWeightedAxisOrientationExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(DistWeightedEccentricityExact, DistWeightedEccentricityExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MaxSquaredDist, MaxSquaredDist);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(MaxSquaredDistExact, MaxSquaredDistExact);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(FilledArea, FilledArea);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(FilledCentroidRow, FilledCentroidRow);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(FilledCentroidColumn, FilledCentroidColumn);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(FilledLengthMajorAxis, FilledLengthMajorAxis);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(FilledLengthMinorAxis, FilledLengthMinorAxis);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(FilledAxisOrientation, FilledAxisOrientation);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(FilledEccentricity, FilledEccentricity);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(FilledInertia, FilledInertia);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(HoleAreaFraction, HoleAreaFraction);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(FilledCentroidDisplacementNormalized, FilledCentroidDisplacementNormalized);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(FilledCompactness, FilledCompactness);
    MTLEARN_FROM_BACKEND_ATTRIBUTE(FilledCircularity, FilledCircularity);
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

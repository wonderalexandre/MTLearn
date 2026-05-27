#pragma once

// Tensor extraction utilities for Connected Filter Preprocessing (CFP).
//
// This is internal C++ implementation, not installed public API. It converts
// the morphology facade tree into Torch tensors consumed by the Python CFP
// autograd functions. The bindings in mtlearn/bindings/cfp should remain thin:
// all tree traversal and tensor-shape knowledge lives here.

#include "mtlearn/detail/morphology_backend.hpp"
#include "mtlearn/morphology.hpp"

#include <cstdint>
#include <list>
#include <memory>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

namespace mtlearn::cfp {

// Builds the tensor payloads needed by the explicit and implicit CFP Jacobian
// paths. The class is stateless because each method derives its output from a
// WeightedTree snapshot supplied by the caller.
class ConnectedFilterPreprocessingTreeTensors {
public:
    // Return one residue per backend internal node slot. Only live node slots
    // carry meaningful values; downstream code should keep using backend node
    // ids/traversal metadata to decide which slots participate in computation.
    static torch::Tensor getResidues(morphology::WeightedTreePtr weightedTree)
    {
        if (!weightedTree) {
            throw std::invalid_argument("Invalid WeightedMorphologicalTree");
        }

        const auto& tree = morphology::detail::topology(*weightedTree);
        const int numNodes = tree.getNumInternalNodeSlots();
        float* residues = new float[numNodes];

        // Only live nodes have meaningful residues. The array keeps backend
        // node ids as direct tensor indices so Python code can index without a
        // remapping table.
        for (morphology::NodeId nodeId : tree.getAliveNodeIds()) {
            residues[nodeId] = morphology::residue(*weightedTree, nodeId);
        }

        return toTensor(residues, numNodes);
    }

    // Materialize the dense logical Jacobian as a sparse COO tensor. Rows are
    // tree node ids and columns are pixel ids. A non-zero entry means that the
    // pixel belongs to the connected component represented by the row node.
    static torch::Tensor getJacobian(morphology::WeightedTreePtr weightedTree)
    {
        if (!weightedTree) {
            throw std::invalid_argument("Invalid WeightedMorphologicalTree");
        }

        const auto& tree = morphology::detail::topology(*weightedTree);
        std::vector<int64_t> rowIndices;
        std::vector<int64_t> colIndices;
        const auto imageSize = tree.getNumRowsOfImage() * tree.getNumColsOfImage();

        // A node contributes to every pixel covered by its complete subtree.
        // The explicit-Jacobian implementation uses this matrix to reconstruct
        // pixels from filtered node residues.
        for (morphology::NodeId nodeId : tree.getAliveNodeIds()) {
            for (morphology::NodeId subtreeNodeId : tree.getNodeSubtree(nodeId)) {
                for (int pixel : tree.getProperParts(subtreeNodeId)) {
                    rowIndices.push_back(nodeId);
                    colIndices.push_back(pixel);
                }
            }
        }

        return toSparseCooTensor(
            rowIndices,
            colIndices,
            tree.getNumInternalNodeSlots(),
            imageSize
        );
    }

    // Return the compact tensor set used by the implicit-Jacobian CFP path:
    // residues, preorder time, postorder time, parent id, and node-of-pixel.
    // This avoids building a potentially large explicit sparse matrix during
    // training and allows the Python autograd function to traverse the tree.
    static std::list<torch::Tensor> getInfoForJacobian(morphology::WeightedTreePtr weightedTree)
    {
        if (!weightedTree) {
            throw std::invalid_argument("Invalid WeightedMorphologicalTree");
        }
        const auto& tree = morphology::detail::topology(*weightedTree);
        if (tree.getNumNodes() == 0) {
            throw std::runtime_error("WeightedMorphologicalTree is empty.");
        }

        const int numNodes = tree.getNumInternalNodeSlots();
        const int numPixels = tree.getNumRowsOfImage() * tree.getNumColsOfImage();

        auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
        auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);

        torch::Tensor tResiduos = torch::zeros({numNodes}, opts_f32);
        torch::Tensor tPreOrder = torch::zeros({numNodes}, opts_i64);
        torch::Tensor tPostOrder = torch::zeros({numNodes}, opts_i64);
        torch::Tensor tParent = torch::zeros({numNodes}, opts_i64);
        torch::Tensor tNodeOfPixel = torch::zeros({numPixels}, opts_i64);

        float* residuesPtr = tResiduos.data_ptr<float>();
        int64_t* preOrderPtr = tPreOrder.data_ptr<int64_t>();
        int64_t* postOrderPtr = tPostOrder.data_ptr<int64_t>();
        int64_t* parentPtr = tParent.data_ptr<int64_t>();
        int64_t* nodeOfPixelPtr = tNodeOfPixel.data_ptr<int64_t>();

        // Node-local metadata is independent per node, so the backend tree can
        // be scanned in parallel. Values for inactive slots keep their zero
        // defaults and are ignored by the traversal orders.
        #pragma omp parallel for
        for (morphology::NodeId nodeId = 0; nodeId < numNodes; ++nodeId) {
            if (tree.isAlive(nodeId)) {
                residuesPtr[nodeId] = morphology::residue(*weightedTree, nodeId);
                preOrderPtr[nodeId] = static_cast<int64_t>(tree.getNodeTimePreOrder(nodeId));
                postOrderPtr[nodeId] = static_cast<int64_t>(tree.getNodeTimePostOrder(nodeId));
                parentPtr[nodeId] = static_cast<int64_t>(tree.getNodeParent(nodeId));
            }
        }

        // nodeOfPixel is the final gather map used to reconstruct image pixels
        // from node-level filtered residues.
        for (int pixel = 0; pixel < numPixels; ++pixel) {
            nodeOfPixelPtr[pixel] = static_cast<int64_t>(tree.getProperPartOwner(pixel));
        }

        std::list<torch::Tensor> result;
        result.push_back(tResiduos);
        result.push_back(tPreOrder);
        result.push_back(tPostOrder);
        result.push_back(tParent);
        result.push_back(tNodeOfPixel);
        return result;
    }

private:
    // Transfer ownership of a raw array into a Torch tensor. The custom deleter
    // captures a shared_ptr so the buffer survives until Torch releases the
    // tensor storage.
    static torch::Tensor toTensor(float* data, int size)
    {
        std::shared_ptr<float> dataPtr(data, [](float* ptr) { delete[] ptr; });
        return torch::from_blob(
            dataPtr.get(),
            {size},
            [dataPtr](void*) mutable {
                // Keep the buffer alive until Python releases the tensor.
            },
            torch::kFloat32);
    }

    // Build a sparse COO tensor from row/column index vectors. Torch copies the
    // stacked index tensors into sparse tensor storage before the local vectors
    // leave scope.
    static torch::Tensor toSparseCooTensor(
        std::vector<int64_t> rowIndices,
        std::vector<int64_t> colIndices,
        int64_t numRows,
        int64_t numCols,
        torch::Dtype dtype = torch::kFloat32)
    {
        torch::Tensor row = torch::from_blob(
            rowIndices.data(),
            {static_cast<long>(rowIndices.size())},
            torch::kInt64);
        torch::Tensor col = torch::from_blob(
            colIndices.data(),
            {static_cast<long>(colIndices.size())},
            torch::kInt64);

        torch::Tensor indices = torch::stack({row, col});
        torch::Tensor values = torch::ones(
            {static_cast<int64_t>(rowIndices.size())},
            torch::TensorOptions().dtype(dtype));

        std::vector<int64_t> size = {numRows, numCols};
        return torch::sparse_coo_tensor(indices, values, size, torch::TensorOptions().dtype(dtype));
    }
};

} // namespace mtlearn::cfp

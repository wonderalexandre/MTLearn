#pragma once

// Tree-traversal implementation for Connected Filter Preprocessing (CFP).
//
// This internal C++ helper implements the CPU tree traversal path used by the
// Python CFP layer variant. It stays outside pybind so the binding layer only
// exposes methods, while this file owns the reconstruction and gradient
// traversal algorithms tied to the morphology tree structure.

#include "mtlearn/detail/morphology_backend.hpp"
#include "mtlearn/morphology.hpp"

#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <torch/torch.h>

namespace mtlearn::cfp {

using morphology::NodeId;
using morphology::WeightedTreePtr;
using TreeTopology = morphology::detail::TreeTopology;

// Stateless operations over a WeightedTree and Torch tensors. Tensor devices
// and dtypes are expected to match the Python caller's CPU traversal path.
class ConnectedFilterPreprocessingTreeTraversal {
public:
    // Reconstruct an image from node-level sigmoid gates.
    //
    // The root starts at its altitude. Each child accumulates its parent's
    // filtered level plus its own residue contribution. Proper parts then
    // scatter the final node levels back to image pixels.
    static torch::Tensor filtering(WeightedTreePtr weightedTree, torch::Tensor sigmoid)
    {
        if (!weightedTree) {
            throw std::invalid_argument("Invalid ValuedMorphologicalTree");
        }

        const TreeTopology& tree = morphology::detail::topology(*weightedTree);
        int numRows = tree.numRows();
        int numCols = tree.numColumns();
        float* sigmoid_ptr = sigmoid.data_ptr<float>();

        std::unique_ptr<float[]> mapLevel(new float[tree.numInternalNodeSlots()]);

        // The root is always kept, and all descendants are expressed as
        // accumulated residue offsets relative to their parent.
        NodeId rootId = tree.root();
        mapLevel[rootId] = morphology::altitude(*weightedTree, rootId) * sigmoid_ptr[rootId];
        for (NodeId nodeId : tree.breadthFirstTraversal()) {
            if (!tree.isRoot(nodeId)) {
                NodeId parentId = tree.parent(nodeId);
                float residue = morphology::residue(*weightedTree, nodeId);
                mapLevel[nodeId] = mapLevel[parentId] + (residue * sigmoid_ptr[nodeId]);
            }
        }

        auto out = torch::empty({numRows, numCols}, torch::kFloat32);
        float* imgOutput = out.data_ptr<float>();
        // Each pixel belongs to exactly one proper part. Writing only proper
        // parts avoids overwriting pixels repeatedly for ancestor nodes.
        for (NodeId nodeId : tree.aliveNodeIds()) {
            for (int pixel : tree.properPart(nodeId)) {
                imgOutput[pixel] = mapLevel[nodeId];
            }
        }

        return out;
    }

    // Compute gradients with respect to the linear CFP parameters used to
    // produce the sigmoid gates. This mirrors the historical CPU tree traversal
    // implementation and is kept separate from the GPU-friendly implicit
    // Python autograd path.
    static std::tuple<torch::Tensor, torch::Tensor> gradients(
        WeightedTreePtr weightedTree,
        torch::Tensor attrs,
        torch::Tensor sigmoid,
        torch::Tensor gradientOfLoss)
    {
        if (!weightedTree) {
            throw std::invalid_argument("Invalid ValuedMorphologicalTree");
        }

        float* attributes = attrs.data_ptr<float>();
        float* sigmoid_ptr = sigmoid.data_ptr<float>();
        const TreeTopology& tree = morphology::detail::topology(*weightedTree);
        int rows = attrs.size(0); // numNodes
        int cols = attrs.size(1); // numFeatures

        // Per-node local derivatives are stored in backend node-id order so
        // the later post-order accumulation can index them directly.
        torch::Tensor gradFilterWeights = torch::empty({rows * cols}, torch::kFloat32);
        torch::Tensor gradFilterBias = torch::empty({rows}, torch::kFloat32);
        float* gradFilterWeights_ptr = gradFilterWeights.data_ptr<float>();
        float* gradFilterBias_ptr = gradFilterBias.data_ptr<float>();

        for (NodeId nodeId : tree.aliveNodeIds()) {
            float dSigmoid = sigmoid_ptr[nodeId] * (1 - sigmoid_ptr[nodeId]);
            float residue = morphology::residue(*weightedTree, nodeId);

            // Compute the local gradient for each node.
            float localDerivative = residue * dSigmoid;

            // Compute the filter gradients for weights and bias.
            gradFilterBias_ptr[nodeId] = localDerivative;
            for (int j = 0; j < cols; j++) {
                gradFilterWeights_ptr[nodeId * cols + j] = localDerivative * attributes[nodeId * cols + j];
            }
        }

        torch::Tensor gradWeight = torch::zeros({cols}, torch::kFloat32);
        torch::Tensor gradBias = torch::zeros({1}, torch::kFloat32);

        float* gradWeight_ptr = gradWeight.data_ptr<float>();
        float* gradBias_ptr = gradBias.data_ptr<float>();
        float* gradLoss = gradientOfLoss.data_ptr<float>();

        // Sum loss gradients for each connected component and accumulate them
        // with the local derivative. Post-order traversal is required because
        // every ancestor receives contributions from all descendant pixels.
        std::unique_ptr<float[]> summationGrad_ptr(new float[tree.numInternalNodeSlots()]);
        morphology::detail::traversePostOrder(
            tree,
            tree.root(),
            [&](NodeId nodeId) -> void {
                summationGrad_ptr[nodeId] = 0;
                for (int p : tree.properPart(nodeId)) {
                    summationGrad_ptr[nodeId] += gradLoss[p];
                }
            },
            [&summationGrad_ptr](NodeId parent, NodeId child) -> void {
                summationGrad_ptr[parent] += summationGrad_ptr[child];
            },
            [&](NodeId nodeId) -> void {
                gradBias_ptr[0] += summationGrad_ptr[nodeId] * gradFilterBias_ptr[nodeId];
                for (int j = 0; j < cols; j++) {
                    gradWeight_ptr[j] += summationGrad_ptr[nodeId] * gradFilterWeights_ptr[nodeId * cols + j];
                }
            });

        return std::make_tuple(gradWeight, gradBias);
    }
};

} // namespace mtlearn::cfp

#include "glomap/estimators/MAC/solvers/mac.h"

#include "glomap/estimators/MAC/solvers/iterative_mac.h"
#include "glomap/estimators/MAC/solvers/mac_utils.h"

#include <Eigen/Dense>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ios>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace glomap {

using mac_internal::AddToLaplacian;
using mac_internal::ComputeFiedler;
using mac_internal::FrankWolfeMac;
using mac_internal::MacEdge;
using mac_internal::TopK;
using mac_internal::UnionFind;
using mac_internal::WeightedLaplacian;

namespace {

struct MacGraph {
  int num_nodes = 0;
  std::vector<image_t> node_to_image;  // local index -> image id
  std::vector<MacEdge> edges;
  int num_outside = 0;  // valid edges outside the largest component
};

// The graph is built after glomap's relative pose estimation, so `inliers`
// holds glomap's own inlier count. The match count is only a fallback.
double EdgeWeight(const ImagePair& pair) {
  if (!pair.inliers.empty()) return static_cast<double>(pair.inliers.size());
  if (pair.matches.rows() > 0) return static_cast<double>(pair.matches.rows());
  return 1.0;
}

inline uint64_t EdgeKey(int a, int b) {
  if (a > b) std::swap(a, b);
  return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(b));
}

// Order-independent FNV-1a fingerprint of a set of pair ids.
uint64_t Fingerprint(std::vector<image_pair_t> ids) {
  std::sort(ids.begin(), ids.end());
  uint64_t h = 1469598103934665603ULL;
  for (image_pair_t id : ids) {
    const uint64_t v = static_cast<uint64_t>(id);
    for (int b = 0; b < 8; ++b) {
      h ^= (v >> (8 * b)) & 0xFF;
      h *= 1099511628211ULL;
    }
  }
  return h;
}

// ===========================================================================
// Step A: graph
// ===========================================================================
MacGraph BuildGraph(const ViewGraph& view_graph,
                    const std::unordered_map<image_t, Image>& images) {
  std::vector<image_pair_t> pair_ids;
  for (const auto& [pair_id, pair] : view_graph.image_pairs) {
    if (!pair.is_valid || pair.image_id1 == pair.image_id2) continue;
    if (images.find(pair.image_id1) == images.end() ||
        images.find(pair.image_id2) == images.end()) {
      continue;
    }
    pair_ids.push_back(pair_id);
  }
  std::sort(pair_ids.begin(), pair_ids.end());

  std::unordered_map<image_t, int> image_to_idx;
  std::vector<image_t> idx_to_image;
  auto get_idx = [&](image_t id) {
    auto it = image_to_idx.find(id);
    if (it != image_to_idx.end()) return it->second;
    const int idx = static_cast<int>(idx_to_image.size());
    image_to_idx.emplace(id, idx);
    idx_to_image.push_back(id);
    return idx;
  };

  std::vector<MacEdge> all_edges;
  for (image_pair_t pair_id : pair_ids) {
    const ImagePair& pair = view_graph.image_pairs.at(pair_id);
    all_edges.push_back({pair_id,
                         get_idx(pair.image_id1),
                         get_idx(pair.image_id2),
                         EdgeWeight(pair)});
  }

  MacGraph graph;
  const int n_all = static_cast<int>(idx_to_image.size());
  if (n_all == 0) return graph;

  UnionFind uf(n_all);
  for (const auto& e : all_edges) uf.Union(e.u, e.v);
  std::vector<int> comp_size(n_all, 0);
  for (int i = 0; i < n_all; ++i) ++comp_size[uf.Find(i)];
  const int best_root = static_cast<int>(
      std::max_element(comp_size.begin(), comp_size.end()) -
      comp_size.begin());

  std::vector<int> to_local(n_all, -1);
  for (int i = 0; i < n_all; ++i) {
    if (uf.Find(i) == best_root) {
      to_local[i] = static_cast<int>(graph.node_to_image.size());
      graph.node_to_image.push_back(idx_to_image[i]);
    }
  }
  graph.num_nodes = static_cast<int>(graph.node_to_image.size());

  for (const auto& e : all_edges) {
    const int lu = to_local[e.u];
    const int lv = to_local[e.v];
    if (lu < 0 || lv < 0) {
      ++graph.num_outside;
      continue;
    }
    graph.edges.push_back({e.pair_id, lu, lv, e.weight});
  }
  return graph;
}

// ===========================================================================
// Step B: backbone (maximum spanning tree + optional odometry chain)
// HEURISTIC HOOK: to try another backbone, change this function only.
// ===========================================================================
std::vector<char> BuildBackbone(const MacGraph& graph,
                                const MacOptions& options,
                                MacStats& stats) {
  const int n = graph.num_nodes;
  const auto& edges = graph.edges;
  std::vector<char> is_backbone(edges.size(), 0);

  std::vector<int> order(edges.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return edges[a].weight > edges[b].weight;
  });
  UnionFind uf(n);
  for (int idx : order) {
    if (uf.Union(edges[idx].u, edges[idx].v)) {
      is_backbone[idx] = 1;
      ++stats.num_tree_edges;
    }
  }

  if (options.use_odometry_backbone) {
    std::unordered_map<uint64_t, int> key_to_edge;
    for (int i = 0; i < static_cast<int>(edges.size()); ++i) {
      key_to_edge[EdgeKey(edges[i].u, edges[i].v)] = i;
    }
    std::vector<int> nodes(n);
    std::iota(nodes.begin(), nodes.end(), 0);
    std::sort(nodes.begin(), nodes.end(), [&](int a, int b) {
      return graph.node_to_image[a] < graph.node_to_image[b];
    });
    for (int i = 0; i + 1 < n; ++i) {
      auto it = key_to_edge.find(EdgeKey(nodes[i], nodes[i + 1]));
      if (it == key_to_edge.end() || is_backbone[it->second]) continue;
      is_backbone[it->second] = 1;
      ++stats.num_odometry_edges;
    }
  }
  return is_backbone;
}

// ===========================================================================
// Step C: MAC (one solve)
// HEURISTIC HOOK: to try another selection rule, write a function with this
// signature and call it in Step C below.
// ===========================================================================
std::vector<int> SelectEdgesMac(const Eigen::MatrixXd& L_base,
                                const std::vector<MacEdge>& candidates,
                                int K,
                                const MacOptions& options,
                                MacStats& stats) {
  const int m = static_cast<int>(candidates.size());
  if (K <= 0) return {};
  if (K >= m) {
    std::vector<int> all(m);
    std::iota(all.begin(), all.end(), 0);
    return all;
  }
  const Eigen::VectorXd w0 =
      Eigen::VectorXd::Constant(m, static_cast<double>(K) / m);
  const Eigen::VectorXd w = FrankWolfeMac(L_base,
                                          candidates,
                                          K,
                                          w0,
                                          options.max_iterations,
                                          options.gap_tolerance,
                                          &stats.fw_iterations);
  stats.lambda2_relaxed =
      ComputeFiedler(WeightedLaplacian(L_base, candidates, w), nullptr);
  return TopK(w, K);  // rounding
}

std::string FormatRounds(const IterativeMacInfo& info) {
  std::ostringstream out;
  out << std::fixed;
  out.precision(3);
  for (const auto& r : info.rounds) {
    out << "\n[MAC]   round " << r.round << ": +" << r.committed_this_round
        << " edges (total " << r.committed_total << ", from "
        << r.candidates << " candidates, " << r.fw_iterations
        << " FW iters), lambda2 " << r.lambda2_after << ", " << r.seconds
        << " s";
  }
  return out.str();
}

}  // namespace

const char* MacAlgorithmName(MacAlgorithm algorithm) {
  switch (algorithm) {
    case MacAlgorithm::MAC:
      return "mac";
    case MacAlgorithm::ITERATIVE:
      return "iterative";
    case MacAlgorithm::ITERATIVE_MST:
      return "iterative_mst";
  }
  return "unknown";
}

// ===========================================================================
// Entry point: graph -> (backbone) -> selection -> apply
// ===========================================================================
MacStats RunMacViewGraphSparsification(
    ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    const MacOptions& options) {
  MacStats stats;

  // ---- Step A: graph ----
  const MacGraph graph = BuildGraph(view_graph, images);
  const int n = graph.num_nodes;
  const int m = static_cast<int>(graph.edges.size());
  stats.num_nodes = n;
  stats.num_valid_edges = m;
  stats.num_outside = graph.num_outside;
  if (n < 3 || m == 0) {
    LOG(INFO) << "[MAC] Graph too small for sparsification, skipping";
    return stats;
  }
  if (n > 5000) {
    LOG(WARNING) << "[MAC] " << n << " nodes: dense eigen-decomposition will "
                 << "be slow.";
  }
  {
    std::vector<image_pair_t> ids;
    for (const auto& e : graph.edges) ids.push_back(e.pair_id);
    stats.graph_hash = Fingerprint(ids);
  }

  IterativeMacOptions iter_options;
  iter_options.fix_fraction = options.fix_fraction;
  iter_options.max_iterations = options.max_iterations;
  iter_options.gap_tolerance = options.gap_tolerance;
  IterativeMacInfo iter_info;

  const double ratio = std::clamp(options.retention_ratio, 0.0, 1.0);
  std::vector<char> keep(m, 0);  // over graph.edges

  if (options.algorithm == MacAlgorithm::ITERATIVE_MST) {
    // ---- Algorithm 2: no backbone; same total number of edges:
    //      (n - 1) + ratio x (m - (n - 1)). ----
    stats.num_candidates = std::max(0, m - (n - 1));
    const int extra = static_cast<int>(std::lround(ratio * stats.num_candidates));
    const int K_total = std::min(m, (n - 1) + extra);
    const std::vector<int> selected =
        IterativeMacWithMst(n, graph.edges, K_total, iter_options, &iter_info);
    for (int idx : selected) keep[idx] = 1;
    stats.num_selected = static_cast<int>(selected.size());
    stats.num_final_tree = iter_info.num_tree + iter_info.num_fill;
  } else {
    // ---- Step B: backbone ----
    const std::vector<char> is_backbone = BuildBackbone(graph, options, stats);

    std::vector<image_pair_t> backbone_ids;
    Eigen::MatrixXd L_base = Eigen::MatrixXd::Zero(n, n);
    std::vector<MacEdge> candidates;
    std::vector<int> cand_to_edge;
    for (int i = 0; i < m; ++i) {
      const MacEdge& e = graph.edges[i];
      if (is_backbone[i]) {
        AddToLaplacian(L_base, e.u, e.v, e.weight);
        backbone_ids.push_back(e.pair_id);
        keep[i] = 1;
      } else {
        candidates.push_back(e);
        cand_to_edge.push_back(i);
      }
    }
    stats.num_backbone_edges = static_cast<int>(backbone_ids.size());
    stats.num_candidates = static_cast<int>(candidates.size());
    stats.backbone_hash = Fingerprint(backbone_ids);
    stats.lambda2_backbone = ComputeFiedler(L_base, nullptr);

    // ---- Step C: selection ----
    const int K = std::clamp(
        static_cast<int>(std::lround(ratio * candidates.size())),
        0,
        static_cast<int>(candidates.size()));

    std::vector<int> selected;
    if (options.algorithm == MacAlgorithm::ITERATIVE) {
      selected = IterativeMacWithFixed(
          L_base, candidates, K, iter_options, &iter_info);
    } else {
      selected = SelectEdgesMac(L_base, candidates, K, options, stats);
    }
    for (int idx : selected) keep[cand_to_edge[idx]] = 1;
    stats.num_selected = static_cast<int>(selected.size());
  }
  stats.num_rounds = static_cast<int>(iter_info.rounds.size());

  // ---- Apply: switch off every edge that is not kept ----
  Eigen::MatrixXd L_final = Eigen::MatrixXd::Zero(n, n);
  int kept = 0;
  for (int i = 0; i < m; ++i) {
    const MacEdge& e = graph.edges[i];
    if (keep[i]) {
      AddToLaplacian(L_final, e.u, e.v, e.weight);
      ++kept;
    } else {
      view_graph.image_pairs.at(e.pair_id).is_valid = false;
      ++stats.num_pruned;
    }
  }
  stats.lambda2_final = ComputeFiedler(L_final, nullptr);
  if (options.algorithm == MacAlgorithm::MAC &&
      (stats.num_selected == 0 || stats.num_selected == stats.num_candidates)) {
    stats.lambda2_relaxed = stats.lambda2_final;  // no optimization needed
  }

  // ---- Log ----
  std::ostringstream log;
  log << "\n[MAC] ========================================"
      << "\n[MAC] Algorithm:              " << MacAlgorithmName(options.algorithm)
      << "\n[MAC] Graph"
      << "\n[MAC]   Images:               " << stats.num_nodes
      << "\n[MAC]   Valid edges:          " << stats.num_valid_edges
      << "\n[MAC]   Outside largest CC:   " << stats.num_outside
      << " (untouched)"
      << "\n[MAC]   Fingerprint:          " << std::hex << stats.graph_hash
      << std::dec;
  if (options.algorithm == MacAlgorithm::ITERATIVE_MST) {
    log << "\n[MAC] Backbone:               none (tree built at the end)";
  } else {
    log << "\n[MAC] Backbone"
        << "\n[MAC]   Spanning tree:        " << stats.num_tree_edges
        << "\n[MAC]   Odometry (extra):     " << stats.num_odometry_edges
        << "\n[MAC]   Total:                " << stats.num_backbone_edges
        << "\n[MAC]   Fingerprint:          " << std::hex
        << stats.backbone_hash << std::dec
        << "\n[MAC]   lambda2:              " << stats.lambda2_backbone;
  }
  log << "\n[MAC] Selection (retention ratio " << ratio << ")";
  if (options.algorithm == MacAlgorithm::MAC) {
    log << "\n[MAC]   Candidates kept:      " << stats.num_selected << " / "
        << stats.num_candidates
        << "\n[MAC]   Frank-Wolfe iters:    " << stats.fw_iterations
        << "\n[MAC]   lambda2 relaxed:      " << stats.lambda2_relaxed;
  } else {
    log << "\n[MAC]   Fix fraction:         " << options.fix_fraction
        << "\n[MAC]   Rounds:               " << stats.num_rounds
        << FormatRounds(iter_info)
        << "\n[MAC]   Committed by rounds:  " << iter_info.num_committed;
    if (options.algorithm == MacAlgorithm::ITERATIVE_MST) {
      log << "\n[MAC]   Final tree:           " << iter_info.num_tree
          << " (+" << iter_info.num_fill << " fill)";
    }
  }
  log << "\n[MAC] Result"
      << "\n[MAC]   Edges kept:           " << kept << " / "
      << stats.num_valid_edges << " ("
      << 100.0 * kept / std::max(1, stats.num_valid_edges) << "%)"
      << "\n[MAC]   Edges pruned:         " << stats.num_pruned
      << "\n[MAC]   lambda2 final:        " << stats.lambda2_final
      << "\n[MAC] ========================================";
  LOG(INFO) << log.str();

  return stats;
}

}  // namespace glomap
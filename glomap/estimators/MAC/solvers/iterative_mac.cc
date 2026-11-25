#include "glomap/estimators/MAC/solvers/iterative_mac.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>

namespace glomap {

using mac_internal::AddToLaplacian;
using mac_internal::ComputeFiedler;
using mac_internal::FrankWolfeMac;
using mac_internal::GreedyInit;
using mac_internal::MacEdge;
using mac_internal::TopK;
using mac_internal::UnionFind;

namespace {

// Shared core of Algorithms 2 and 3: commit `target` edges from `edges` in
// rounds. `L_fixed` starts as the Laplacian of the fixed set and is updated
// in place with every committed edge. `committed[i]` marks committed edges.
// `last_unrounded` receives the unrounded weight of every edge that was a
// candidate in the LAST round (others stay 0); Algorithm 2 uses it.
void CommitInRounds(Eigen::MatrixXd& L_fixed,
                    const std::vector<MacEdge>& edges,
                    int target,
                    const IterativeMacOptions& options,
                    std::vector<char>& committed,
                    Eigen::VectorXd* last_unrounded,
                    IterativeMacInfo* info) {
  const int m = static_cast<int>(edges.size());
  target = std::min(target, m);
  int num_committed = 0;
  int round = 0;

  while (num_committed < target) {
    ++round;
    const auto t0 = std::chrono::steady_clock::now();

    // How many edges to commit this round: a fraction of what is left.
    const int remaining = target - num_committed;
    int fix_now = static_cast<int>(remaining * options.fix_fraction);
    fix_now = std::clamp(fix_now, 1, remaining);

    // Candidates: every edge not committed yet.
    std::vector<int> cand_idx;
    std::vector<MacEdge> cand_edges;
    for (int i = 0; i < m; ++i) {
      if (!committed[i]) {
        cand_idx.push_back(i);
        cand_edges.push_back(edges[i]);
      }
    }

    // Run MAC for fix_now edges, starting from the greedy (strongest) set.
    int fw_iterations = 0;
    Eigen::VectorXd unrounded;
    if (fix_now >= static_cast<int>(cand_edges.size())) {
      unrounded = Eigen::VectorXd::Ones(cand_edges.size());
    } else {
      unrounded = FrankWolfeMac(L_fixed,
                                cand_edges,
                                fix_now,
                                GreedyInit(cand_edges, fix_now),
                                options.max_iterations,
                                options.gap_tolerance,
                                &fw_iterations);
    }

    if (last_unrounded) {
      last_unrounded->setZero();
      for (size_t k = 0; k < cand_idx.size(); ++k) {
        (*last_unrounded)(cand_idx[k]) = unrounded(k);
      }
    }

    // Commit the fix_now edges with the highest unrounded weight; they are
    // part of the fixed set from the next round on.
    for (int li : TopK(unrounded, fix_now)) {
      const int gi = cand_idx[li];
      committed[gi] = 1;
      AddToLaplacian(L_fixed, edges[gi].u, edges[gi].v, edges[gi].weight);
      ++num_committed;
    }

    if (info) {
      IterativeMacRound r;
      r.round = round;
      r.committed_this_round = fix_now;
      r.committed_total = num_committed;
      r.candidates = static_cast<int>(cand_edges.size());
      r.fw_iterations = fw_iterations;
      r.lambda2_after = ComputeFiedler(L_fixed, nullptr);
      r.seconds = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
      info->rounds.push_back(r);
    }
  }
  if (info) info->num_committed = num_committed;
}

}  // namespace

// ---------------------------------------------------------------------------
// Algorithm 3: iterative MAC with a fixed edge set (the backbone).
// ---------------------------------------------------------------------------
std::vector<int> IterativeMacWithFixed(const Eigen::MatrixXd& L_fixed,
                                       const std::vector<MacEdge>& candidates,
                                       int K,
                                       const IterativeMacOptions& options,
                                       IterativeMacInfo* info) {
  const int m = static_cast<int>(candidates.size());
  K = std::clamp(K, 0, m);

  std::vector<char> committed(m, 0);
  Eigen::MatrixXd L = L_fixed;
  CommitInRounds(L, candidates, K, options, committed, nullptr, info);

  std::vector<int> selected;
  for (int i = 0; i < m; ++i) {
    if (committed[i]) selected.push_back(i);
  }
  return selected;
}

// ---------------------------------------------------------------------------
// Algorithm 2: iterative MAC without a backbone, spanning tree at the end.
// ---------------------------------------------------------------------------
std::vector<int> IterativeMacWithMst(int num_nodes,
                                     const std::vector<MacEdge>& edges,
                                     int K_total,
                                     const IterativeMacOptions& options,
                                     IterativeMacInfo* info) {
  const int m = static_cast<int>(edges.size());
  const int tree_slots = num_nodes - 1;
  K_total = std::clamp(K_total, 0, m);
  const int num_fix_target = K_total - tree_slots;

  std::vector<char> committed(m, 0);
  Eigen::VectorXd last_unrounded = Eigen::VectorXd::Zero(m);

  if (num_fix_target > 0) {
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(num_nodes, num_nodes);
    CommitInRounds(
        L, edges, num_fix_target, options, committed, &last_unrounded, info);
  } else {
    // Budget fits in a tree: no MAC rounds, just a spanning tree. The
    // Python code uses equal weights here; we use inlier counts instead.
    for (int i = 0; i < m; ++i) last_unrounded(i) = edges[i].weight;
  }

  // Final spanning tree: committed edges first (like weight = inf in the
  // Python code), then the remaining edges by last unrounded weight.
  UnionFind uf(num_nodes);
  for (int i = 0; i < m; ++i) {
    if (committed[i]) uf.Union(edges[i].u, edges[i].v);
  }

  std::vector<int> rest;
  for (int i = 0; i < m; ++i) {
    if (!committed[i]) rest.push_back(i);
  }
  std::stable_sort(rest.begin(), rest.end(), [&](int a, int b) {
    return last_unrounded(a) > last_unrounded(b);
  });

  std::vector<char> in_tree(m, 0);
  int num_tree = 0;
  for (int i : rest) {
    if (num_tree >= tree_slots) break;
    if (uf.Union(edges[i].u, edges[i].v)) {
      in_tree[i] = 1;
      ++num_tree;
    }
  }

  // If the tree used fewer than n - 1 slots (the committed edges already
  // connected a lot), fill the rest with the best remaining edges.
  int num_fill = 0;
  for (int i : rest) {
    if (num_tree + num_fill >= tree_slots) break;
    if (!in_tree[i]) {
      in_tree[i] = 1;
      ++num_fill;
    }
  }

  if (info) {
    info->num_tree = num_tree;
    info->num_fill = num_fill;
  }

  std::vector<int> selected;
  for (int i = 0; i < m; ++i) {
    if (committed[i] || in_tree[i]) selected.push_back(i);
  }
  return selected;
}

}  // namespace glomap
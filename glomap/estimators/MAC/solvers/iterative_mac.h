#pragma once

// Iterative MAC, ported from algo2_3.py.
//
// Algorithm 3 (IterativeMacWithFixed): keeps a fixed edge set (our backbone)
//   and fills the budget in rounds. Each round runs MAC for a fraction of the
//   remaining budget, commits the edges with the highest unrounded weight,
//   and adds them to the fixed set for the next round.
//
// Algorithm 2 (IterativeMacWithMst): no backbone up front. Commits
//   K_total - (n - 1) edges in rounds the same way, then builds a spanning
//   tree from the remaining edges (ordered by MAC's last unrounded weights)
//   to guarantee connectivity, filling any unused tree slots with the best
//   remaining edges.

#include "glomap/estimators/MAC/solvers/mac_utils.h"

#include <Eigen/Dense>

#include <vector>

namespace glomap {

struct IterativeMacOptions {
  // Fraction of the remaining budget committed per round (at least 1 edge).
  // 0.3 matches fix_per_iter_frac in algo2_3.py.
  double fix_fraction = 0.3;

  // Frank-Wolfe settings for each round.
  int max_iterations = 20;
  double gap_tolerance = 1e-6;
};

struct IterativeMacRound {
  int round = 0;
  int committed_this_round = 0;
  int committed_total = 0;
  int candidates = 0;       // candidates offered to MAC in this round
  int fw_iterations = 0;
  double lambda2_after = 0.0;  // lambda_2 of the fixed set after committing
  double seconds = 0.0;
};

struct IterativeMacInfo {
  std::vector<IterativeMacRound> rounds;
  int num_committed = 0;  // edges committed by the MAC rounds
  int num_tree = 0;       // Algorithm 2 only: edges added by the final tree
  int num_fill = 0;       // Algorithm 2 only: unused tree slots filled
};

// Algorithm 3. Returns indices into `candidates` of the K kept edges.
std::vector<int> IterativeMacWithFixed(
    const Eigen::MatrixXd& L_fixed,
    const std::vector<mac_internal::MacEdge>& candidates,
    int K,
    const IterativeMacOptions& options,
    IterativeMacInfo* info);

// Algorithm 2. Returns indices into `edges` of the K_total kept edges
// (committed edges + spanning tree), on a graph with `num_nodes` nodes.
std::vector<int> IterativeMacWithMst(
    int num_nodes,
    const std::vector<mac_internal::MacEdge>& edges,
    int K_total,
    const IterativeMacOptions& options,
    IterativeMacInfo* info);

}  // namespace glomap
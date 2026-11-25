#pragma once

// Shared building blocks for mac.cc and iterative_mac.cc.

#include "glomap/scene/image.h"

#include <Eigen/Dense>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace glomap {
namespace mac_internal {

struct UnionFind {
  std::vector<int> parent;
  std::vector<int> rank;

  explicit UnionFind(int n) : parent(n), rank(n, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int Find(int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }

  bool Union(int a, int b) {
    a = Find(a);
    b = Find(b);
    if (a == b) return false;
    if (rank[a] < rank[b]) std::swap(a, b);
    parent[b] = a;
    if (rank[a] == rank[b]) ++rank[a];
    return true;
  }
};

struct MacEdge {
  image_pair_t pair_id;
  int u;          // local node index
  int v;          // local node index
  double weight;  // inlier count
};

inline void AddToLaplacian(Eigen::MatrixXd& L, int u, int v, double w) {
  L(u, u) += w;
  L(v, v) += w;
  L(u, v) -= w;
  L(v, u) -= w;
}

// lambda_2 of L and (optionally) its Fiedler vector. Dense O(N^3).
inline double ComputeFiedler(const Eigen::MatrixXd& L,
                             Eigen::VectorXd* fiedler) {
  if (L.rows() < 2) return 0.0;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(L);
  if (es.info() != Eigen::Success) {
    LOG(WARNING) << "[MAC] Eigen decomposition failed";
    if (fiedler) *fiedler = Eigen::VectorXd::Zero(L.rows());
    return 0.0;
  }
  if (fiedler) *fiedler = es.eigenvectors().col(1);
  return std::max(0.0, es.eigenvalues()(1));
}

// Indices of the k largest entries (ties broken by index -> deterministic).
inline std::vector<int> TopK(const Eigen::VectorXd& values, int k) {
  std::vector<int> idx(values.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
                   [&](int a, int b) { return values(a) > values(b); });
  idx.resize(std::min<size_t>(std::max(k, 0), idx.size()));
  return idx;
}

inline Eigen::MatrixXd WeightedLaplacian(
    const Eigen::MatrixXd& L_base,
    const std::vector<MacEdge>& candidates,
    const Eigen::VectorXd& w) {
  Eigen::MatrixXd L = L_base;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (w(i) <= 0.0) continue;
    AddToLaplacian(
        L, candidates[i].u, candidates[i].v, w(i) * candidates[i].weight);
  }
  return L;
}

// Indicator of the K strongest candidates (NaiveGreedy in the Python code).
inline Eigen::VectorXd GreedyInit(const std::vector<MacEdge>& candidates,
                                  int K) {
  Eigen::VectorXd weights(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    weights(i) = candidates[i].weight;
  }
  Eigen::VectorXd w = Eigen::VectorXd::Zero(candidates.size());
  for (int idx : TopK(weights, K)) w(idx) = 1.0;
  return w;
}

// Frank-Wolfe on the relaxed MAC problem:
//   maximize lambda_2(L_base + sum_i w_i * L_i)
//   s.t. w in [0,1]^m, sum(w) = K
// Starts from w_init (must be feasible) and returns the UNROUNDED weights.
inline Eigen::VectorXd FrankWolfeMac(const Eigen::MatrixXd& L_base,
                                     const std::vector<MacEdge>& candidates,
                                     int K,
                                     Eigen::VectorXd w,
                                     int max_iterations,
                                     double gap_tolerance,
                                     int* iterations_out = nullptr) {
  const int m = static_cast<int>(candidates.size());
  Eigen::VectorXd grad(m);
  Eigen::VectorXd fiedler;

  int it = 0;
  for (; it < max_iterations; ++it) {
    const double lambda2 =
        ComputeFiedler(WeightedLaplacian(L_base, candidates, w), &fiedler);

    // Gradient of lambda_2 w.r.t. w_i: weight_i * (f_u - f_v)^2
    for (int i = 0; i < m; ++i) {
      const double d = fiedler(candidates[i].u) - fiedler(candidates[i].v);
      grad(i) = candidates[i].weight * d * d;
    }

    // Best vertex of the feasible set: all K units on the largest gradients.
    Eigen::VectorXd s = Eigen::VectorXd::Zero(m);
    for (int idx : TopK(grad, K)) s(idx) = 1.0;

    const double gap = grad.dot(s - w);
    if (gap <= gap_tolerance * std::max(1.0, lambda2)) break;

    w += (2.0 / (it + 2.0)) * (s - w);
  }
  if (iterations_out) *iterations_out = it;
  return w;
}

}  // namespace mac_internal
}  // namespace glomap
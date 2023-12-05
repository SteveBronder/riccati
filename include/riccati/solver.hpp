#ifndef INCLUDE_riccati_SOLVER_HPP
#define INCLUDE_riccati_SOLVER_HPP

#include <riccati/chebyshev.hpp>
#include <riccati/utils.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <complex>
#include <cmath>
#include <vector>
// REMOVE THIS
#include <iostream>

namespace riccati {

namespace internal {
  inline Eigen::VectorXd logspace(double start, double end, int num, double base) {
    Eigen::VectorXd result(num);
    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num; ++i) {
        result[i] = std::pow(base, start + i * delta);
    }

    return result;
}
}
// OmegaFun / GammaFun take in a scalar and return a complex<Scalar>
template <typename OmegaFun, typename GammaFun, typename Scalar_,
          typename Integral_>
class SolverInfo {
 public:
  using Scalar = Scalar_;
  using Integral = Integral_;
  using complex_t = std::complex<Scalar>;
  using matrixd_t = matrix_t<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  using vectord_t = vector_t<Scalar>;
  // Frequency function
  OmegaFun omega_fun_;
  // Friction function
  GammaFun gamma_fun_;
  /**
   * Current state vector of size (2,), containing the numerical solution and
   * its derivative.
   */
  Eigen::Matrix<std::complex<Scalar>, 2, 1> y_;
  // Frequency and friction functions evaluated at n+1 Chebyshev nodes
  vectord_t omega_n_;
  vectord_t gamma_n_;
  // idk yet
  vectorc_t un_;
  std::pair<complex_t, complex_t> a_;
  // Number of nodes and diff matrices
  Integral n_nodes_{0};
  // Differentiation matrices and Vectors of Chebyshev nodes
  std::vector<std::pair<matrixd_t, vectord_t>> chebyshev_;
  matrixd_t Dn_;
  // Lengths of node vectors
  vectord_t ns_;
  /**
   * Values of the independent variable evaluated at (`n` + 1) Chebyshev
   * nodes over the interval [x, x + `h`], where x is the value of the
   * independent variable at the start of the current step and `h` is the
   * current stepsize.
   */
  vectord_t xn_;
  /**
   * Values of the independent variable evaluated at (`p` + 1) Chebyshev
   * nodes over the interval [x, x + `h`], where x is the value of the
   * independent variable at the start of the current step and `h` is the
   * current stepsize.
   */
  vectord_t xp_;
  /**
   * Values of the independent variable evaluated at `p` points
   * over the interval [x, x + `h`] lying in between Chebyshev nodes, where x is
   * the value of the independent variable at the start of the current step and
   * `h` is the current stepsize. The in-between points :math:`\\tilde{x}_p` are
   * defined by
   * $$
   * \\tilde{x}_p = \cos\left( \\frac{(2k + 1)\pi}{2p} \\right), \quad k = 0, 1,
   * \ldots p-1.
   * $$
   */
  vectord_t xp_interp_;
  vectord_t yn_;
  vectord_t dyn_;

  /**
   * Interpolation matrix of size (`p`+1, `p`), used for interpolating
   * function between the nodes `xp` and `xpinterp` (for computing Riccati
   * stepsizes).
   */
  matrixd_t L_;
  // Clenshaw-Curtis quadrature weights
  vectord_t quadwts_;

  matrixd_t integration_matrix_;
  /**
   * Minimum and maximum (number of Chebyshev nodes - 1) to use inside
   * Chebyshev collocation steps. The step will use `nmax` nodes or the
   * minimum number of nodes necessary to achieve the required local error,
   * whichever is smaller. If `nmax` > 2`nini`, collocation steps will be
   * attempted with :math:`2^i` `nini` nodes at the ith iteration.
   */
  Integral nini_;
  Integral nmax_;
  // (Number of Chebyshev nodes - 1) to use for computing Riccati steps.
  Integral n_;
  // (Number of Chebyshev nodes - 1) to use for estimating Riccati stepsizes.
  Integral p_;
  /**
   * Initial interval length which will be used to estimate the initial
   * derivatives of w, g.
   */
  Scalar h0_;
  Scalar h_;  // Current stepsize
  // Counters for various operations
  Integral n_chebnodes_{0};
  // Number of Chebyshev steps attempted.
  Integral n_chebstep_{0};
  /**
   * Number of times an iteration of the Chebyshev-grid-based
   * collocation method has been performed (note that if `nmax` >=
   * 4`nini` then a single Chebyshev step may include multiple
   * iterations!).
   */
  Integral n_chebits_{0};
  // Number of times a linear system has been solved.
  Integral n_LS_{0};
  /**
   * Number of times an iteration of the Chebyshev-grid-based
   * collocation method has been performed (note that if `nmax` >=
   * 4`nini` then a single Chebyshev step may include multiple iterations!).
   */
  Integral n_riccstep_{0};
  bool denseout_{false};  // Dense output flag

  inline auto build_chebyshev(Integral nini, Integral n_nodes) {
    std::vector<std::pair<matrixd_t, vectord_t>> res(n_nodes + 1, std::make_pair(matrixd_t{}, vectord_t{}));
    // Compute Chebyshev nodes and differentiation matrices
    for (Integral i = 0; i <= n_nodes_; ++i) {
      res[i] = chebyshev<Scalar>(nini * std::pow(2, i));
    }
    return res;
  }
  // Constructor
  template <typename OmegaFun_, typename GammaFun_>
  SolverInfo(OmegaFun_&& omega_fun, GammaFun_&& gamma_fun, Scalar h0, Integral nini,
             Integral nmax, Integral n, Integral p, bool dense_output)
      : omega_fun_(std::forward<OmegaFun_>(omega_fun)),
        gamma_fun_(std::forward<GammaFun_>(gamma_fun)),
        y_(Eigen::Matrix<std::complex<Scalar>, 2, 1>::Zero()),
        omega_n_(n + 1),
        gamma_n_(n + 1),
        n_nodes_(log2(nmax / nini) + 1),
        chebyshev_(build_chebyshev(nini, n_nodes_)),
        ns_(internal::logspace(std::log2(nini), std::log2(nini) + n_nodes_ - 1, n_nodes_, 2.0)),
        xn_(),
        xp_(),
        xp_interp_(),
        L_(),
        quadwts_(quad_weights<Scalar>(n)),
        integration_matrix_(dense_output ? integration_matrix<Scalar>(n + 1)
                                         : matrixd_t(0, 0)),
        nini_(nini),
        nmax_(nmax),
        n_(n),
        p_(p),
        h0_(h0),
        h_(h0),
        n_chebnodes_(n_nodes_),
        n_chebstep_(n_nodes_),
        n_chebits_(n_nodes_),
        n_LS_(n_nodes_),
        n_riccstep_(n_nodes_),
        denseout_(dense_output) {

    // Set Dn and xn if available
    auto it = std::find(ns_.begin(), ns_.end(), n);
    if (it != ns_.end()) {
      Integral idx = std::distance(ns_.begin(), it);
      Dn_ = chebyshev_[idx].first;
      xn_ = chebyshev_[idx].second;
    } else {
      std::tie(Dn_, xn_) = chebyshev<Scalar>(n);
    }

    // Set xp and xpinterp
    if (std::find(ns_.begin(), ns_.end(), p) != ns_.end()) {
      xp_ = chebyshev_[std::distance(ns_.begin(),
                                 std::find(ns_.begin(), ns_.end(), p))].second;
    } else {
      xp_ = chebyshev<Scalar>(p).second;
    }
    xp_interp_ = (vector_t<Scalar>::LinSpaced(p_, pi<Scalar>() / (2.0 * p_),
      pi<Scalar>() * (1.0 - (1.0 / (2.0 * p_)))).array()).cos().matrix();
    L_ = interpolate(xp_, xp_interp_);  // Assuming interp is a function that
                                       // creates the interpolation matrix
  }

  // Method to increase various counters
  void increase(int chebnodes = 0, int chebstep = 0, int chebits = 0,
                int LS = 0, int riccstep = 0) {
    n_chebnodes_ += chebnodes;
    n_chebstep_ += chebstep;
    n_chebits_ += chebits;
    n_LS_ += LS;
    n_riccstep_ += riccstep;
  }
  std::tuple<Integral, Integral, Integral, Integral, Integral> output(
      const std::vector<Integral>& steptypes) {
    Integral cheb_steps = 0;
    Integral ricc_steps = 0;
    const auto stepcount_size = steptypes.size();
    for (std::size_t i = 0; i < stepcount_size; ++i) {
      cheb_steps += steptypes[i];
      ricc_steps += !steptypes[i];
    }
    return {n_chebstep_, cheb_steps, n_chebits_, n_LS_, n_chebnodes_};
  }
};

template <bool DenseOutput, typename OmegaFun, typename GammaFun, typename Scalar,
          typename Integral>
inline auto make_solver(OmegaFun&& omega_fun, GammaFun&& gamma_fun, Scalar h0, Integral nini,
                        Integral nmax, Integral n, Integral p) {
  return SolverInfo<std::decay_t<OmegaFun>, std::decay_t<GammaFun>, Scalar,
                    Integral>(std::forward<OmegaFun>(omega_fun),
                              std::forward<GammaFun>(gamma_fun), h0, nini, nmax, n,
                              p, DenseOutput);
}

}  // namespace riccati

#endif

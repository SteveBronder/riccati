#ifndef INCLUDE_riccati_SOLVER_HPP
#define INCLUDE_riccati_SOLVER_HPP

#include <riccati/utils.hpp>
#include <riccati/chebyshev.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <vector>

namespace riccati {
// OmegaFun / GammaFun take in a scalar and return a complex<Scalar>
template <typename OmegaFun, typename GammaFun, typename Scalar_,
          typename Integral_>
class Solverinfo {
 public:
  using complex_t = std::complex<Scalar>;
  using matrixc_t = matrix_t<complex_t>;
  using matrixd_t = matrix_t<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  using vectord_t = vector_t<Scalar>;
  using Scalar = Scalar_;
  using Integral = Integral_;
  // Frequency function
  OmegaFun omega_fun_;
  // Friction function
  GammaFun gamma_fun_;
  /**
   * Current state vector of size (2,), containing the numerical solution and
   * its derivative.
   */
  vectorc_t y_;
  // Frequency and friction functions evaluated at n+1 Chebyshev nodes
  vectorc_t omega_n_;
  vectorc_t gamma_n_;
  // Number of nodes and diff matrices
  Integral n_nodes_{0};
  // Differentiation matrices and Vectors of Chebyshev nodes
  std::vector<std::pair<matrixd_t, vectord_t>> chebyshev_;
  matrixd_t Dn_;
  vectord_t xn_;
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
  vectord_t xpinterp_;
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

  static auto build_chebyshev(Integral n_nodes) {
    std::vector<std::pair<matrixd_t, vectord_t>> chebyshev(
        n_nodes, std::make_pair(matrixd_t{}, vectord_t{}));
    // Compute Chebyshev nodes and differentiation matrices
    for (unsigned int i = 0; i <= n_nodes_; ++i) {
      chebyshev[i] = chebyshev(current_n);
    }
    return chebyshev;
  }
  // Constructor
  Solverinfo(OmegaFun omega_fun, GammaFun gamma_fun, Scalar h0, Integral nini,
             Integral nmax, Integral n, Integral p, bool dense_output)
      : omega_fun_(omega_fun),
        gamma_fun_(gamma_fun),
        y_(2, vectorc_t::Zero()),
        omega_n_(n + 1),
        gamma_n_(n + 1),
        n_nodes_(log2(nmax / nini) + 1),
        diff_matrices_(n_nodes_),
        nodes_(n_nodes_),
        ns_(n_nodes_),
        xn_(),
        xp_(),
        xpinterp_(),
        L_(),
        quadwts_(quad_weights(n)),
        integration_matrix_(dense_output ? integration_matrix(n + 1)
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
    ns_ = vectord_t::LinSpaced(n_nodes_, nini_,
                               nini_ * std::pow(2, Dlength - 1));

    // Set Dn and xn if available
    auto it = std::find(ns_.begin(), ns_.end(), n + 1);
    if (it != ns_.end()) {
      Integral idx = std::distance(ns_.begin(), it);
      Dn_ = diff_matrices_[idx];
      xn_ = nodes_[idx];
    } else {
      std::tie(Dn_, xn_) = chebyshev(n);
    }

    // Set xp and xpinterp
    if (std::find(ns_.begin(), ns_.end(), p + 1) != ns_.end()) {
      xp_ = nodes_[std::distance(ns_.begin(),
                                 std::find(ns_.begin(), ns_.end(), p + 1))];
    } else {
      xp_ = chebyshev(p).second;
    }
    xpinterp_.resize(p);
    for (Integral k = 0; k < p; ++k) {
      xpinterp_(k) = cos((2 * k + 1) * M_PI / (2 * p));
    }
    L_ = integration(xp_, xpinterp_);  // Assuming interp is a function that
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
    const auto stepcount_size
        = steptypes.size() for (std::size_t i = 0; i < stepcount_size; ++i) {
      cheb_steps += steptypes[i];
      ricc_steps += !steptypes[i];
    }
    return {n_chebstep_, cheb_steps, n_chebits_, n_LS_, n_chebnodes_};
  }
};

template <typename OmegaFun, typename GammaFun, typename Scalar,
          typename Integral>
inline auto make_solver(OmegaFun&& omega_fun, GammaFun&& gamma_fun, Scalar nini,
                        Integral nmax, Integral n, Integral p) {
  return SolverInfo<std::decay_t<OmegaFun>, std::decay_t<GammaFun>, Scalar,
                    Integral>(std::forward<OmegaFun>(omega_fun),
                              std::forward<GammaFun>(gamma_fun), nini, nmax, n,
                              p);
}

}  // namespace riccati

#endif

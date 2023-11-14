#ifndef INCLUDE_riccati_EVOLVE_HPP
#define INCLUDE_riccati_EVOLVE_HPP

#include <complex>
#include <riccati/utils.hpp>
#include <type_traits>

namespace riccati {
template <typename SolverInfo, typename Scalar, typename Y0>
auto osc_evolve(SolverInfo &&info, Scalar x0, Scalar x1, Scalar h, Y0 y0,
                Scalar epsres, Scalar epsh) {
  auto sign = std::sign(h);
  if (sign * (x0 + h) > sign * x1) {
    h = x1 - x0;
    auto xscaled = h / 2.0 * info.xn + xc0 + h / 2.0;
    info.wn = info.w(xscaled);
    info.gn = info.g(xscaled);
  }
  info.h = h;
}

/**
 * @brief Solves the differential equation y'' + 2gy' + w^2y = 0 over a given
 * interval.
 *
 * This function solves the differential equation on the interval (xi, xf),
 * starting from the initial conditions y(xi) = yi and y'(xi) = dyi. It keeps
 * the residual of the ODE below eps, and returns an interpolated solution
 * (dense output) at the points specified in xeval.
 *
 * @tparam SolverInfo Type of the solver info object containing differentiation
 * matrices, etc.
 * @tparam Scalar Numeric scalar type, typically float or double.
 * @tparam Vec Type of the vector for dense output values, should match Scalar
 * type.
 *
 * @param[in] info SolverInfo object containing necessary information for the
 * solver.
 * @param[in] xi Starting value of the independent variable.
 * @param[in] xf Ending value of the independent variable.
 * @param[in] yi Initial value of the dependent variable at xi.
 * @param[in] dyi Initial derivative of the dependent variable at xi.
 * @param[in] eps Relative tolerance for the local error of both Riccati and
 * Chebyshev type steps.
 * @param[in] epsh Relative tolerance for choosing the stepsize of Riccati
 * steps.
 * @param[in] xeval List of x-values where the solution is to be interpolated
 * (dense output) and returned.
 * @param[in] hard_stop If true, forces the solver to have a potentially smaller
 * last stepsize to stop exactly at xf.
 * @param[in] warn If true, displays warnings during the run; otherwise,
 * warnings are silenced.
 *
 * @return Tuple containing:
 *         - Vector of x-values at internal steps of the solver (xs).
 *         - Vector of y-values of the dependent variable at internal steps
 * (ys).
 *         - Vector of derivatives of the dependent variable at internal steps
 * (dys).
 *         - Vector indicating the success of each step (1 for success, 0 for
 * failure).
 *         - Vector of complex phases accumulated during each successful Riccati
 * step (phases).
 *         - Vector indicating the types of successful steps taken (1 for
 * Riccati, 0 for Chebyshev).
 *         - Vector of interpolated values of the solution at xeval (yeval).
 */
template <typename SolverInfo, typename Scalar, typename Vec>
auto solve(SolverInfo &&info, Scalar xi, Scalar xf, std::complex<Scalar> yi,
           std::complex<Scalar> dyi, Scalar eps, Scalar epsh, Vec &&xeval,
           bool hard_stop = false, bool warn = false) {
  using complex_t = std::complex<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  using vectord_t = vector_t<Scalar>;

  Scalar intdir = (xf - xi > 0) ? 1 : -1;
  if (intdir * info.h0 < 0) {
    throw std::domain_error(
        "Direction of itegration does not match stepsize sign,"
        " adjusting it so that integration happens from xi to xf.");
  }
  // Check that yeval and xeval are right size
  if (info.denseout) {
    if (!xeval.size()) {
      throw std::domain_error("Dense output requrested but xeval is size 0!");
    }
    // TODO: Better error messages
    const bool high_range_err = intdir * xf > (intdir * xeval.maxCoeff());
    const bool low_range_err = intdir * xf > (intdir * xeval.minCoeff());
    if (high_range_err || low_range_err) {
      if (high_range_err && low_range_err) {
        throw std::domain_error(
            "Some dense output points lie outside the integration range!")
      }
      if (high_range_err) {
        throw std::domain_error(
            "Some dense output points lie outside the integration range!")
      }
      if (low_range_err) {
        throw std::domain_error(
            "Some dense output points lie outside the integration range!")
      }
    }
  }

  // Initialize vectors for storing results
  vectord_t xs(100);
  vectorc_t ys(100);
  vectorc_t dys(100);
  std::vector<int> successes;
  std::vector<int> steptypes;
  vectorc_t phases;
  vectorc_t yeval(xeval.size());

  complex_t y = yi;
  complex_t dy = dyi;
  complex_t yprev = y;
  complex_t dyprev = dy;
  auto scale_xi = scale(xn, xi, hi).eval();
  auto omega_is = info.omega_fun(scale_xi).eval();
  auto gamma_is = info.omega_fun(scale_xi).eval();
  auto wi = wis.mean();
  auto gi = gis.mean();
  auto dwi = (2.0 / hi * (info.Dn * wis)).mean();
  auto dgi = (2.0 / hi * (info.Dn * wis)).mean();

  auto hslo_ini = intdir * std::min(1e8, std::abs(1.0 / wi));
  auto hosc_ini = intdir
                  * std::min(std::min(1e8, std::abs(wi / dwi)),
                             std::abs(gi / dgi)) if (hard_stop) {
    hosc_ini = (intdir * (xi + hosc_ini) > intdir * xf) ? xf - xi : hosc_ini;
    hslo_ini = (intdir * (xi + hslo_ini) > intdir * xf) ? xf - xi : hslo_ini;
  }
  auto hslo = choose_nonosc_stepsize(info.omega_fun_, xi, hslo_ini);
  auto hosc = choose_osc_stepsize(info.omega_fun_, info.gamma_fun_, info.p_,
                                  info.n_, info.xp_, info.xp_interp_, info.xn_,
                                  info.L_, xi, hosc_ini, epsh);
  Scalar xcurrent = xi;
  auto wnext = wi;
  auto dwnext = dwi;
  auto dense_positions = vectord_t(info.denseout ? xeval.size() : 0);
  while (std::abs(xcurrent - xf) > Scalar(1e-8)
         && intdir * xcurrent < intdir * xf) {
    bool success = false;
    bool steptype = true;
    if ((intdir * hosc > intdir * hslo * 5.0)
        && (intdir * hosc * wnext / (2.0 * pi()) > 1.0)) {
      if (hard_stop) {
        if (int dir *(xcurrent + hosc) > intdir * xf) {
          hosc = xf - xcurrent;
          auto xscaled = (xcurrent + hosc / 2.0 + hosc / 2.0 * info.xp).eval();
          info.wn = info.omega_fun(xscaled);
          info.gn = info.gamma_fun(xscaled);
        }
        if (intdir * (xcurrent + hslo) > intdir * xf) {
          hslo = xf - xcurrent;
        }
      }
      std::tie(y, dy, res, success, phase)
          = osc_step(info, xcurrent, hosc, yprev, dyprev, eps);
      steptype = 1;
    }
    while (!success) {
      std::tie(y, dy, err, success) = nonosc_step(info, xcurrent, hslo, yprev, dyprev, eps);
      phase = 0;
      steptype = 0;
      if (!success) {
        hslo *= 0.5;
      }
      if (intdir * hslo < 1e-16) {
        throw std::domain_error("Stepsize became to small error");
      }
    }
    auto h = steptype ? hosc : hslo;
    if (info.denseout){
        std::size_t dense_size = 0;
        std::size_t dense_start = 0;
        vectord_t xdense(x_eval.size());
        // Assuming xeval is sorted we just want start and size
        {
          unsigned int i = 0
          for (; i < dense_positions.size(); ++i) {
              if ((intdir * xeval[i] >= intdir * xcurrent &&
                  intdir * xeval[i] < intdir * (xcurrent + h)) ||
                  (xeval[i] == xf && xeval[i] == (xcurrent + h))) {
                    dense_start = i;
                    dense_size++;
                    break;
                  }
          }
          for (; i < dense_positions.size(); ++i) {
              if ((intdir * xeval[i] >= intdir * xcurrent &&
                  intdir * xeval[i] < intdir * (xcurrent + h)) ||
                  (xeval[i] == xf && xeval[i] == (xcurrent + h))) {
                    dense_size++;
                  } else {
                    break;
                  }
          }
        }
        //auto xdense = xeval[positions]
        auto x_eval_map = Eigen::Map<vectord_t>(xeval.data() + dense_start, dense_size);
        auto y_eval_map = Eigen::Map<vectorc_t>(yeval.data() + dense_start, dense_size);
        if (steptype) {
            // xscaled = xcurrent + h/2 + h/2*info.xn
            auto xscaled = 2.0 / h * (x_eval_map - xcurrent) - 1.0;
            auto Linterp = interp(info.xn_, xscaled);
            auto udense = Linterp * info.un_;
            auto fdense = udense.exp();
            y_eval_map = info.a_[0] * fdense + info.a_[1] * fdense.conj();
        } else {
            auto xscaled = xcurrent + h / 2 + h / 2 * info.nodes[1];
            auto Linterp = interp(xscaled, xdense);
            y_eval_map = Linterp * info.yn_;
        }
    }
  }

  return std::make_tuple(xs, ys, dys, successes, phases, steptypes, yeval);
}

}  // namespace riccati

#endif

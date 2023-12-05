#ifndef INCLUDE_riccati_EVOLVE_HPP
#define INCLUDE_riccati_EVOLVE_HPP

#include <riccati/solver.hpp>
#include <riccati/step.hpp>
#include <riccati/stepsize.hpp>
#include <riccati/utils.hpp>
#include <complex>
#include <type_traits>



namespace riccati {
template <typename SolverInfo, typename Scalar, typename Y0>
auto osc_evolve(SolverInfo &&info, Scalar x0, Scalar x1, Scalar h, Y0 y0,
                Scalar epsres, Scalar epsilon_h) {
  auto sign = std::signbit(h) ? -1 : 1;
  if (sign * (x0 + h) > sign * x1) {
    h = x1 - x0;
    auto xscaled = h / 2.0 * info.xn + x0 + h / 2.0;
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
 * (dense output) at the points specified in x_eval.
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
 * @param[in] epsilon_h Relative tolerance for choosing the stepsize of Riccati
 * steps.
 * @param[in] x_eval List of x-values where the solution is to be interpolated
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
 *         - Vector of interpolated values of the solution at x_eval (yeval).
 */
template <typename SolverInfo, typename Scalar, typename Vec>
auto solve(SolverInfo &&info, Scalar xi, Scalar xf, std::complex<Scalar> yi,
           std::complex<Scalar> dyi, Scalar eps, Scalar epsilon_h, Vec &&x_eval,
           bool hard_stop = false, bool warn = false) {
  using complex_t = std::complex<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  using matrixc_t = matrix_t<complex_t>;
  using vectord_t = vector_t<Scalar>;
  using stdvecd_t = std::vector<Scalar>;
  using stdvecc_t = std::vector<complex_t>;
  Scalar intdir = std::signbit(info.h0_) ? -1 : 1;
  if (intdir * (xf - xi) < 0) {
    throw std::domain_error(
        "Direction of integration does not match stepsize sign,"
        " adjusting it so that integration happens from xi to xf.");
  }
  // Check that yeval and x_eval are right size
  if (info.denseout_) {
    if (!x_eval.size()) {
      throw std::domain_error("Dense output requested but x_eval is size 0!");
    }
    // TODO: Better error messages
    const bool high_range_err = intdir * xf > (intdir * x_eval.maxCoeff());
    const bool low_range_err = intdir * xi > (intdir * x_eval.minCoeff());
    if (high_range_err || low_range_err) {
      if (high_range_err && low_range_err) {
        throw std::domain_error(
            "Some dense output points lie outside the high and low of the integration range!");
      }
      if (high_range_err) {
        throw std::domain_error(
            "Some dense output points lie outside the high of the integration range!");
      }
      if (low_range_err) {
        throw std::domain_error(
            "Some dense output points lie outside the low of the integration range!");
      }
    }
  }

  // Initialize vectors for storing results
  std::size_t output_size = 100;
  stdvecd_t xs;
  xs.reserve(output_size);
  stdvecc_t ys;
  ys.reserve(output_size);
  stdvecc_t dys;
  dys.reserve(output_size);
  std::vector<int> successes;
  successes.reserve(output_size);
  std::vector<int> steptypes;
  steptypes.reserve(output_size);
  stdvecd_t phases;
  phases.reserve(output_size);
  vectorc_t yeval(x_eval.size());

  complex_t y = yi;
  complex_t dy = dyi;
  complex_t yprev = y;
  complex_t dyprev = dy;
  auto scale_xi = (xi + info.h0_ / 2.0 + info.h0_ / 2.0 * info.xn_.array()).matrix().eval();
  auto omega_is = info.omega_fun_(scale_xi).eval();
  auto gamma_is = info.gamma_fun_(scale_xi).eval();
  Scalar wi = omega_is.mean();
  Scalar gi = gamma_is.mean();
  Scalar dwi = (2.0 / info.h0_ * (info.Dn_ * omega_is)).mean();
  Scalar dgi = (2.0 / info.h0_ * (info.Dn_ * gamma_is)).mean();
  Scalar hslo_ini = intdir * std::min(static_cast<Scalar>(1e8), std::abs(1.0 / wi));
  Scalar hosc_ini = intdir
                  * std::min(std::min(static_cast<Scalar>(1e8), std::abs(wi / dwi)),
                             std::abs(gi / dgi));

  if (hard_stop) {
    hosc_ini = (intdir * (xi + hosc_ini) > intdir * xf) ? xf - xi : hosc_ini;
    hslo_ini = (intdir * (xi + hslo_ini) > intdir * xf) ? xf - xi : hslo_ini;
  }
  auto hslo = choose_nonosc_stepsize(info, xi, hslo_ini, 0.2);
  auto hosc = choose_osc_stepsize(info, xi, hosc_ini, epsilon_h);
  Scalar xcurrent = xi;
  Scalar wnext = wi;
  auto dense_positions = vectord_t(info.denseout_ ? x_eval.size() : 0);
  matrixc_t y_eval;
  matrixc_t dy_eval;
  int iter = 0;
  while (std::abs(xcurrent - xf) > Scalar(1e-8)
         && intdir * xcurrent < intdir * xf) {
    std::cout << "iter: " << iter << std::endl;
    iter++;
    Scalar phase{0.0};
    bool success = false;
    bool steptype = true;
    Scalar err;
    if ((intdir * hosc > intdir * hslo * 5.0)
        && (intdir * hosc * wnext / (2.0 * pi<Scalar>()) > 1.0)) {
      if (hard_stop) {
        if (intdir * (xcurrent + hosc) > intdir * xf) {
          hosc = xf - xcurrent;
          auto xscaled = ((xcurrent + hosc / 2.0 + hosc / 2.0 * info.xp_.array()).matrix()).eval();
          info.omega_n_ = info.omega_fun_(xscaled);
          info.gamma_n_ = info.gamma_fun_(xscaled);
        }
        if (intdir * (xcurrent + hslo) > intdir * xf) {
          hslo = xf - xcurrent;
        }
      }
      std::tie(y, dy, err, success, phase)
          = osc_step(info, xcurrent, hosc, yprev, dyprev, eps);
      steptype = 1;
    }
    while (!success) {
      std::tie(y, dy, err, success, y_eval, dy_eval) = nonosc_step(info, xcurrent, hslo, yprev, dyprev, eps);
      steptype = 0;
      if (!success) {
        hslo *= 0.5;
      }
      if (intdir * hslo < 1e-16) {
        throw std::domain_error("Stepsize became to small error");
      }
    }
    auto h = steptype ? hosc : hslo;
    if (info.denseout_){
        Eigen::Index dense_size = 0;
        Eigen::Index dense_start = 0;
        // Assuming x_eval is sorted we just want start and size
        {
          Eigen::Index i = 0;
          for (; i < dense_positions.size(); ++i) {
              if ((intdir * x_eval[i] >= intdir * xcurrent &&
                  intdir * x_eval[i] < intdir * (xcurrent + h)) ||
                  (x_eval[i] == xf && x_eval[i] == (xcurrent + h))) {
                    dense_start = i;
                    dense_size++;
                    break;
                  }
          }
          for (; i < dense_positions.size(); ++i) {
              if ((intdir * x_eval[i] >= intdir * xcurrent &&
                  intdir * x_eval[i] < intdir * (xcurrent + h)) ||
                  (x_eval[i] == xf && x_eval[i] == (xcurrent + h))) {
                    dense_size++;
                  } else {
                    break;
                  }
          }
        }
        auto x_eval_map = Eigen::Map<vectord_t>(x_eval.data() + dense_start, dense_size);
        auto y_eval_map = Eigen::Map<vectorc_t>(yeval.data() + dense_start, dense_size);
        if (steptype) {
            // xscaled = xcurrent + h/2 + h/2*info.xn
            auto xscaled = (2.0 / h * (x_eval_map.array() - xcurrent) - 1.0).matrix().eval();
            auto Linterp = interpolate(info.xn_, xscaled);
            auto udense = Linterp * info.un_;
            auto fdense = udense.array().exp().matrix().eval();
            y_eval_map = info.a_.first * fdense + info.a_.second * fdense.conjugate();
        } else {
            auto xscaled = (xcurrent + h / 2 + h / 2 * info.chebyshev_[1].second.array()).matrix().eval();
            auto Linterp = interpolate(xscaled, x_eval_map);
            y_eval_map = Linterp * y_eval;
        }
    }
    // Finish appending and ending conditions
    ys.push_back(y);
    dys.push_back(dy);
    xs.push_back(xcurrent + h);
    phases.push_back(phase);
    steptypes.push_back(steptype);
    successes.push_back(success);
    Scalar dwnext;
    Scalar gnext;
    Scalar dgnext;
    if (steptype) {
      std::cout << "\tosc update" << std::endl;
      wnext = info.omega_n_[0];
      gnext = info.gamma_n_[0];
      // ERROR HERE
      dwnext = 2.0 / h * info.Dn_.row(0).dot(info.omega_n_);
      dgnext = 2.0 / h * info.Dn_.row(0).dot(info.gamma_n_);
    } else {
      std::cout << "\tnonosc update" << std::endl;
      wnext = info.omega_fun_(xcurrent + h);
      gnext = info.gamma_fun_(xcurrent + h);
      print("Dn: ", info.Dn_);
      print("xn", info.xn_);
      print("h", h);
      auto gam_eval = info.omega_fun_((xcurrent + h / 2.0 + h / 2.0 * info.xn_.array()).matrix()).eval();
      print("gam_eval", gam_eval);
      auto test2 = info.Dn_.row(0).dot(gam_eval);
      print("test2", test2);
      dwnext = 2.0 / h * test2;
      dgnext = 2.0 / h * info.Dn_.row(0).dot(info.gamma_fun_((xcurrent + h / 2.0 + h / 2.0 * info.xn_.array()).matrix()));
    }
    print("\twnext", wnext);
    print("\tdwnext", dwnext);
    print("\tgnext", gnext);
    print("\tdgnext", dgnext);
    print("\txcurrent", xcurrent);
    print("\tintdir", intdir);
    xcurrent += h;
    if (intdir * xcurrent < intdir * xf) {
      hslo_ini = intdir * std::min(1e8, std::abs(1.0 / wnext));
      hosc_ini = intdir * std::min(std::min(1e8, std::abs(wnext / dwnext)), std::abs(gnext / dgnext));
      if (hard_stop) {
        if (intdir * (xcurrent + hosc_ini) > intdir * xf) {
          hosc_ini = xf - xcurrent;
        }
        if (intdir * (xcurrent + hslo_ini) > intdir * xf) {
          hslo_ini = xf - xcurrent;
        }
      }
      hosc = choose_osc_stepsize(info, xcurrent, hosc_ini, epsilon_h);
      hslo = choose_nonosc_stepsize(info, xcurrent, hslo_ini, 0.2);
      print("hosc_ini", hosc_ini);
      print("hslo_ini", hslo_ini);
      print("hosc", hosc);
      print("hslo", hslo);
      yprev = y;
      dyprev = dy;
    }
  }
  return std::make_tuple(std::move(xs), std::move(ys), std::move(dys), std::move(successes),
    std::move(phases), std::move(steptypes), std::move(yeval));
}

}  // namespace riccati

#endif

#ifndef INCLUDE_riccati_EVOLVE_HPP
#define INCLUDE_riccati_EVOLVE_HPP

#include <riccati/chebyshev.hpp>
#include <riccati/step.hpp>
#include <riccati/stepsize.hpp>
#include <riccati/utils.hpp>
#include <complex>
#include <type_traits>
#include <tuple>

namespace riccati {

template <typename SolverInfo, typename Scalar, typename Vec, typename Vec2>
inline auto osc_evolve(SolverInfo &&info, Scalar xi, Scalar xf,
                       std::complex<Scalar> yi, std::complex<Scalar> dyi,
                       Scalar eps, Scalar epsilon_h, Scalar init_stepsize,
                       Vec &&x_eval, Vec2 &&y_true, bool hard_stop = false,
                       bool warn = false) {
  int sign = init_stepsize > 0 ? 1 : -1;
  using complex_t = std::complex<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  using vectord_t = vector_t<Scalar>;
  auto xscaled = (xi + init_stepsize / 2.0 + init_stepsize / 2.0 * info.xn_.array()).matrix().eval();
    info.omega_n_ = info.omega_fun_(xscaled);
    info.gamma_n_ = info.gamma_fun_(xscaled);

  vectorc_t yeval;
  // TODO: Add this check to regular evolve
  if (sign * (xi + init_stepsize) > sign * xf) {
    throw std::out_of_range(
        std::string("Stepsize (") + std::to_string(init_stepsize)
        + std::string(") is too large for integration range (")
        + std::to_string(xi) + std::string(", ") + std::to_string(xf)
        + std::string(")!"));
  }
  auto osc_ret = osc_step(info, xi, init_stepsize, yi, dyi, eps);
  if (std::get<0>(osc_ret) == 0) {
    return std::make_tuple(false, xi, init_stepsize, osc_ret,
                           vectorc_t(0), static_cast<Eigen::Index>(0), static_cast<Eigen::Index>(0));
  } else {
    Eigen::Index dense_size = 0;
    Eigen::Index dense_start = 0;
    if (info.denseout_) {
      // Assuming x_eval is sorted we just want start and size
      std::tie(dense_start, dense_size) = get_slice(x_eval, sign * xi, sign * (xi + init_stepsize));
      if (dense_size != 0) {
        auto x_eval_map = x_eval.segment(dense_start, dense_size);
        auto xscaled = (2.0 / init_stepsize * (x_eval_map.array() - xi) - 1.0)
                            .matrix()
                            .eval();
        auto Linterp = interpolate(info.xn_, xscaled);
        auto udense = (Linterp * info.un_).eval();
        auto fdense = udense.array().exp().matrix().eval();
        yeval = info.a_.first * fdense + info.a_.second * fdense.conjugate();
      }
    }
    auto x_next = xi + init_stepsize;
    auto wnext = info.omega_n_[0];
    auto gnext = info.gamma_n_[0];
    auto dwnext = 2.0 / init_stepsize * info.Dn_.row(0).dot(info.omega_n_);
    auto dgnext = 2.0 / init_stepsize * info.Dn_.row(0).dot(info.gamma_n_);
    auto hosc_ini = sign
                    * std::min(std::min(1e8, std::abs(wnext / dwnext)),
                               std::abs(gnext / dgnext));
    if (sign * (x_next + hosc_ini) > sign * xf) {
      hosc_ini = xf - x_next;
    }
    auto h_next = choose_osc_stepsize(info, x_next, hosc_ini, epsilon_h);
    return std::make_tuple(true, x_next, h_next, osc_ret, yeval, dense_start, dense_size);
  }
}

template <typename SolverInfo, typename Scalar, typename Vec, typename Vec2>
inline auto nonosc_evolve(SolverInfo &&info, Scalar xi, Scalar xf,
                       std::complex<Scalar> yi, std::complex<Scalar> dyi,
                       Scalar eps, Scalar epsilon_h, Scalar init_stepsize,
                       Vec &&x_eval, Vec2 &&y_true, bool hard_stop = false,
                       bool warn = false) {
  int sign = init_stepsize > 0 ? 1 : -1;
  using complex_t = std::complex<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  using vectord_t = vector_t<Scalar>;
  auto xscaled = (xi + init_stepsize / 2.0 + init_stepsize / 2.0 * info.xn_.array()).matrix().eval();
    info.omega_n_ = info.omega_fun_(xscaled);
    info.gamma_n_ = info.gamma_fun_(xscaled);

  vectorc_t yeval;
  // TODO: Add this check to regular evolve
  if (sign * (xi + init_stepsize) > sign * xf) {
    throw std::out_of_range(
        std::string("Stepsize (") + std::to_string(init_stepsize)
        + std::string(") is too large for integration range (")
        + std::to_string(xi) + std::string(", ") + std::to_string(xf)
        + std::string(")!"));
  }
  auto nonosc_ret = nonosc_step(info, xi, init_stepsize, yi, dyi, eps);
  if (!std::get<0>(nonosc_ret)) {
    return std::make_tuple(false, xi, init_stepsize, nonosc_ret,
                           vectorc_t(0), static_cast<Eigen::Index>(0), static_cast<Eigen::Index>(0));
  } else {
    Eigen::Index dense_size = 0;
    Eigen::Index dense_start = 0;
    if (info.denseout_) {
      // Assuming x_eval is sorted we just want start and size
      std::tie(dense_start, dense_size) = get_slice(x_eval, sign * xi, sign * (xi + init_stepsize));
      if (dense_size != 0) {
        auto x_eval_map = x_eval.segment(dense_start, dense_size);
          auto xscaled
              = (xi + init_stepsize / 2 + init_stepsize / 2 * info.chebyshev_[1].second.array())
                    .matrix()
                    .eval();
      auto Linterp = interpolate(xscaled, x_eval_map);
        yeval = Linterp * std::get<4>(nonosc_ret);
      }
    }
    auto x_next = xi + init_stepsize;
    auto wnext = info.omega_fun_(xi + init_stepsize);
    auto gnext = info.gamma_fun_(xi + init_stepsize);
    auto gam_eval
        = info.omega_fun_(
                  (xi + init_stepsize / 2.0 + init_stepsize / 2.0 * info.xn_.array()).matrix())
              .eval();
    auto test2 = info.Dn_.row(0).dot(gam_eval);
    auto dwnext = 2.0 / init_stepsize * test2;
    auto dgnext = 2.0 / init_stepsize
              * info.Dn_.row(0).dot(info.gamma_fun_(
                  (xi + init_stepsize / 2.0 + init_stepsize / 2.0 * info.xn_.array()).matrix()));
    auto hslo_ini = sign * std::min(1e8, std::abs(1.0 / wnext));
    if (sign * (x_next + hslo_ini) > sign * xf) {
      hslo_ini = xf - x_next;
    }
    auto h_next = choose_nonosc_stepsize(info, x_next, hslo_ini, epsilon_h);
    return std::make_tuple(true, x_next, h_next, nonosc_ret, yeval, dense_start, dense_size);
  }
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
template <typename SolverInfo, typename Scalar, typename Vec, typename Vec2>
inline auto evolve(SolverInfo &&info, Scalar xi, Scalar xf,
                   std::complex<Scalar> yi, std::complex<Scalar> dyi,
                   Scalar eps, Scalar epsilon_h, Scalar init_stepsize,
                   Vec &&x_eval, Vec2 &&y_true, bool hard_stop = false,
                   bool warn = false) {
  using complex_t = std::complex<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  using matrixc_t = matrix_t<complex_t>;
  using vectord_t = vector_t<Scalar>;
  using stdvecd_t = std::vector<Scalar>;
  using stdvecc_t = std::vector<complex_t>;
  Scalar intdir = init_stepsize > 0 ? 1 : -1;
  if (init_stepsize * (xf - xi) < 0) {
    throw std::domain_error(
        "Direction of integration does not match stepsize sign,"
        " adjusting it so that integration happens from xi to xf.");
  }
  // Check that yeval and x_eval are right size
  if (info.denseout_) {
    if (!x_eval.size()) {
      throw std::domain_error("Dense output requested but x_eval is size 0!");
    }
    print("x_eval", x_eval);
    // TODO: Better error messages
    auto x_eval_max = (intdir * x_eval.maxCoeff());
    auto x_eval_min = (intdir * x_eval.minCoeff());
    auto xi_intdir = intdir * xi;
    auto xf_intdir = intdir * xf;
    const bool high_range_err = xf_intdir < x_eval_max;
    const bool low_range_err = xi_intdir > x_eval_min;
    if (high_range_err || low_range_err) {
      if (high_range_err && low_range_err) {
        throw std::out_of_range(
            std::string{"The max and min of the output points ("}
            + std::to_string(x_eval_min) + std::string{", "}
            + std::to_string(x_eval_max)
            + ") lie outside the high and low of the integration range ("
            + std::to_string(xi_intdir) + std::string{", "}
            + std::to_string(xf_intdir) + ")!");
      }
      if (high_range_err) {
        throw std::out_of_range(
            std::string{"The max of the output points ("}
            + std::to_string(x_eval_max)
            + ") lies outside the high of the integration range ("
            + std::to_string(xf_intdir) + ")!");
      }
      if (low_range_err) {
        throw std::out_of_range(
            std::string{"The min of the output points ("}
            + std::to_string(x_eval_min)
            + ") lies outside the low of the integration range ("
            + std::to_string(xi_intdir) + ")!");
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
  auto scale_xi
      = (xi + init_stepsize / 2.0 + init_stepsize / 2.0 * info.xn_.array())
            .matrix()
            .eval();
  auto omega_is = info.omega_fun_(scale_xi).eval();
  auto gamma_is = info.gamma_fun_(scale_xi).eval();
  Scalar wi = omega_is.mean();
  Scalar gi = gamma_is.mean();
  Scalar dwi = (2.0 / init_stepsize * (info.Dn_ * omega_is)).mean();
  Scalar dgi = (2.0 / init_stepsize * (info.Dn_ * gamma_is)).mean();
  Scalar hslo_ini
      = intdir * std::min(static_cast<Scalar>(1e8), std::abs(1.0 / wi));
  Scalar hosc_ini
      = intdir
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
  matrixc_t y_eval;
  matrixc_t dy_eval;
  int iter = 0;
  while (std::abs(xcurrent - xf) > Scalar(1e-8)
         && intdir * xcurrent < intdir * xf) {
    std::cout << "\niter: " << iter << std::endl;
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
          auto xscaled
              = ((xcurrent + hosc / 2.0 + hosc / 2.0 * info.xp_.array())
                     .matrix())
                    .eval();
          info.omega_n_ = info.omega_fun_(xscaled);
          info.gamma_n_ = info.gamma_fun_(xscaled);
        }
        if (intdir * (xcurrent + hslo) > intdir * xf) {
          hslo = xf - xcurrent;
        }
      }
      std::tie(success, y, dy, err, phase)
          = osc_step(info, xcurrent, hosc, yprev, dyprev, eps);
      steptype = 1;
    }
    while (!success) {
      std::tie(success, y, dy, err, y_eval, dy_eval)
          = nonosc_step(info, xcurrent, hslo, yprev, dyprev, eps);
      steptype = 0;
      if (!success) {
        hslo *= 0.5;
      }
      if (intdir * hslo < 1e-16) {
        throw std::domain_error("Stepsize became to small error");
      }
    }
    auto h = steptype ? hosc : hslo;
    if (info.denseout_) {
      Eigen::Index dense_size = 0;
      Eigen::Index dense_start = 0;
      // Assuming x_eval is sorted we just want start and size
      std::cout << "xcurrent: " << xcurrent << std::endl;
      std::cout << "intdir * xcurrent: " << intdir * xcurrent << std::endl;
      std::cout << "xcurrent + h: " << xcurrent + h << std::endl;
      std::cout << "intdir * (xcurrent + h): " << intdir * (xcurrent + h)
                << std::endl;
      {
        Eigen::Index i = 0;
        for (; i < x_eval.size(); ++i) {
          if ((intdir * x_eval[i] >= intdir * xcurrent
               && intdir * x_eval[i] < intdir * (xcurrent + h))
              || (x_eval[i] == xf && x_eval[i] == (xcurrent + h))) {
            dense_start = i;
            break;
          }
        }
        for (; i < x_eval.size(); ++i) {
          if ((intdir * x_eval[i] >= intdir * xcurrent
               && intdir * x_eval[i] < intdir * (xcurrent + h))
              || (x_eval[i] == xf && x_eval[i] == (xcurrent + h))) {
            dense_size++;
          } else {
            break;
          }
        }
      }
      if (dense_size != 0) {
        auto x_eval_map
            = Eigen::Map<vectord_t>(x_eval.data() + dense_start, dense_size);
        auto y_eval_map
            = Eigen::Map<vectorc_t>(yeval.data() + dense_start, dense_size);
        auto y_true_map
            = Eigen::Map<vectorc_t>(y_true.data() + dense_start, dense_size);
        std::cout << "Writing from " << dense_start << " to "
                  << dense_start + dense_size << std::endl;
        if (steptype) {
          // xscaled = xcurrent + h/2 + h/2*info.xn
          std::cout << "========osc dense: =======" << std::endl;
          auto xscaled = (2.0 / h * (x_eval_map.array() - xcurrent) - 1.0)
                             .matrix()
                             .eval();
          auto Linterp = interpolate(info.xn_, xscaled);
          auto udense = (Linterp * info.un_).eval();
          auto fdense = udense.array().exp().matrix().eval();
          print("xscaled", xscaled);
          print("Linterp", Linterp);
          print("udense", udense);
          print("fdense", fdense);
          y_eval_map
              = info.a_.first * fdense + info.a_.second * fdense.conjugate();
          print("y_eval_map", vectorc_t(y_eval_map));
          vectord_t y_err
              = ((y_true_map - y_eval_map).array() / y_true_map.array())
                    .abs()
                    .matrix();
          print("y_err", y_err);
        } else {
          std::cout << "=======nonosc dense: ========" << std::endl;
          auto xscaled
              = (xcurrent + h / 2 + h / 2 * info.chebyshev_[1].second.array())
                    .matrix()
                    .eval();
          auto Linterp = interpolate(xscaled, x_eval_map);
          y_eval_map = Linterp * y_eval;
          vectord_t y_err
              = ((y_true_map - y_eval_map).array() / y_true_map.array())
                    .abs()
                    .matrix();
          print("y_err", y_err);
        }
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
      wnext = info.omega_n_[0];
      gnext = info.gamma_n_[0];
      dwnext = 2.0 / h * info.Dn_.row(0).dot(info.omega_n_);
      dgnext = 2.0 / h * info.Dn_.row(0).dot(info.gamma_n_);
    } else {
      wnext = info.omega_fun_(xcurrent + h);
      gnext = info.gamma_fun_(xcurrent + h);
      auto gam_eval
          = info.omega_fun_(
                    (xcurrent + h / 2.0 + h / 2.0 * info.xn_.array()).matrix())
                .eval();
      auto test2 = info.Dn_.row(0).dot(gam_eval);
      dwnext = 2.0 / h * test2;
      dgnext = 2.0 / h
               * info.Dn_.row(0).dot(info.gamma_fun_(
                   (xcurrent + h / 2.0 + h / 2.0 * info.xn_.array()).matrix()));
    }
    xcurrent += h;
    if (intdir * xcurrent < intdir * xf) {
      hslo_ini = intdir * std::min(1e8, std::abs(1.0 / wnext));
      hosc_ini = intdir
                 * std::min(std::min(1e8, std::abs(wnext / dwnext)),
                            std::abs(gnext / dgnext));
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
      yprev = y;
      dyprev = dy;
    }
  }
  return std::make_tuple(xs, ys, dys, successes, phases, steptypes, yeval);
}

}  // namespace riccati

#endif

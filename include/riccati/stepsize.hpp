#ifndef INCLUDE_riccati_STEPSIZE_HPP
#define INCLUDE_riccati_STEPSIZE_HPP

#include <riccati/utils.hpp>

namespace riccati {
/**
 * Chooses the stepsize for spectral Chebyshev steps, based on the variation
 * of 1/w, the approximate timescale over which the solution changes. If over
 *  the suggested interval h 1/w changes by a fraction of :math:`\pm``epsilon_h`
 * or more, the interval is halved, otherwise it's accepted.
 *
 *  @tparam SolverInfo A riccati solver like object
 *  @tparam FloatingPoint A floating point
 *  @param info Solverinfo object which is used to retrieve Solverinfo.xp, the
 * (p+1) Chebyshev nodes used for interpolation to determine the stepsize.
 *  @param x0 Current value of the independent variable.
 *  @param h Initial estimate of the stepsize.
 *  @param epsilon_h Tolerance parameter defining how much 1/w(x) is allowed to
 * change over the course of the step.
 *
 *  @return Refined stepsize over which 1/w(x) does not change by more than
 * epsilon_h/w(x).
 *
 */
template <typename SolverInfo, typename FloatingPoint>
inline FloatingPoint choose_nonosc_stepsize(SolverInfo&& info, FloatingPoint x0,
                                            FloatingPoint h,
                                            FloatingPoint epsilon_h) {
  auto ws = info.omega_fun_(riccati::scale(info.xp_, x0, h));
  if (ws.maxCoeff() > (1 + epsilon_h) / std::abs(h)) {
    return choose_nonosc_stepsize(info, x0, h / 2.0, epsilon_h);
  } else {
    return h;
  }
}

// TODO: Better Names
// TODO: What if h is too small to start with? Recurse forever?
template <typename SolverInfo, typename FloatingPoint>
inline FloatingPoint choose_osc_stepsize(SolverInfo&& info, FloatingPoint x0,
                                         FloatingPoint h,
                                         FloatingPoint epsilon_h) {
  auto t = riccati::scale(info.xp_interp_, x0, h).eval();
  auto s = riccati::scale(info.xp_, x0, h).eval();
  // TODO: Use a memory arena for these
  using vectord_t = vector_t<FloatingPoint>;
  vectord_t ws(s.size());
  vectord_t gs(s.size());
  if (info.p_ == info.n_) {
    info.omega_n_ = info.omega_fun_(s);
    info.gamma_n_ = info.gamma_fun_(s);
    ws = info.omega_n_;
    gs = info.gamma_n_;
  } else {
    vectord_t xn_scaled = riccati::scale(info.xn_, x0, h);
    info.omega_n_ = info.omega_fun_(xn_scaled);
    info.gamma_n_ = info.gamma_fun_(xn_scaled);
    ws = info.omega_fun_(s);
    gs = info.gamma_fun_(s);
  }
  vectord_t omega_analytic = info.omega_fun_(t);
  auto omega_estimate = info.L_ * ws;
  vectord_t gamma_analytic = info.gamma_fun_(t);
  auto gamma_estimate = info.L_ * gs;
  FloatingPoint max_omega_err
      = (((omega_estimate - omega_analytic).array() / omega_analytic.array())
             .abs())
            .maxCoeff();
  FloatingPoint max_gamma_err
      = (((gamma_estimate - gamma_analytic).array() / gamma_analytic.array())
             .abs())
            .maxCoeff();
  FloatingPoint max_err = std::max(max_omega_err, max_gamma_err);
  if (max_err > epsilon_h) {
    auto h_scaling = std::min(
        0.7, 0.9 * std::pow(epsilon_h / max_err, (1.0 / (info.p_ - 1.0))));
    return choose_osc_stepsize(info, x0, h * h_scaling, epsilon_h);
  } else {
    return h;
  }
}

}  // namespace riccati
#endif

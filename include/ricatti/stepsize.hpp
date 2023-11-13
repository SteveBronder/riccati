#ifndef INCLUDE_RICATTI_STEPSIZE_HPP
#define INCLUDE_RICATTI_STEPSIZE_HPP

#include <ricatti/solver.hpp>
#include <ricatti/utils.hpp>

namespace ricatti {
/**
 * Chooses the stepsize for spectral Chebyshev steps, based on the variation
 * of 1/w, the approximate timescale over which the solution changes. If over
 *  the suggested interval h 1/w changes by a fraction of :math:`\pm``epsh` or
 * more, the interval is halved, otherwise it's accepted.
 *
 *  @tparam SolverInfo A ricatti solver like object
 *  @tparam FloatingPoint A floating point
 *  @param info Solverinfo object which is used to retrieve Solverinfo.xp, the (p+1) Chebyshev nodes used for interpolation to determine the stepsize.
 *  @param x0 Current value of the independent variable.
 *  @param h Initial estimate of the stepsize.
 *  @param epsh Tolerance parameter defining how much 1/w(x) is allowed to
 * change over the course of the step.
 *
 *  @return Refined stepsize over which 1/w(x) does not change by more than
 * epsh/w(x).
 *
 */
template <typename Omega_F, typename T_Xp, typename FloatingPoint>
auto choose_nonosc_stepsize(Omega_F&& omega_fun, T_Xp xp, FloatingPoint x0, FloatingPoint epsh) {
  auto ws = omega_fun(ricatti::scale(xp, x0, h));
  if (ws.maxCoeff() > (1 + epsh) / std::abs(h)) {
    return choose_nonosc_stepsize(omega_fun, xp, x0, h / 2.0, epsh);
  } else {
    return h;
  }
}

// TODO: Better Names
// TODO: What if h is too small to start with? Recurse forever?
template <typename Omega_F, typename Gamma_F, typename Xp, typename XpInterp, typename T_P, typename T_N, typename Xn, typename LMat, typename FloatingPoint>
auto choose_nonosc_stepsize(Omega_F&& omega_fun, Gamma_F&& gamma_fun, T_P p, T_N n,
  Xp&& xp, XpInterp&& xp_interp, Xn&& xn, LMat&& L, FloatingPoint x0, FloatingPoint epsh) {
  auto t = ricatti::scale(xp_interp, x0, h);
  auto s = ricatti::scale(xp, x0, h);
  // TODO: Use a memory arena for these
  Eigen::VectorXd wn(s.size());
  Eigen::VectorXd gn(s.size());
  Eigen::VectorXd ws(s.size());
  Eigen::VectorXd gs(s.size());
  if (p == n) {
    wn = omega_fun(s);
    gn = gamma_fun(s);
    ws = wn;
    gs = gn;
  } else {
    auto xn_scaled = ricatti::scale(xn, x0, h);
    wn = omega_fun(xn_scaled);
    gn = gamma_fun(xn_scaled);
    ws = w(s);
    gs = g(s);
  }
  auto omega_analytic = w(t);
  auto omega_estimate = L * ws;
  auto gamma_analytic = g(t);
  auto gamma_estimate = L * gs;
  auto max_omega_err = (((west - wana) / wana).abs()).maxCoeff();
  auto max_gamma_err = (((gest - gana) / gana).abs()).maxCoeff();
  auto max_err = std::max(max_omega_err, max_gamma_err);
  if (max_err > epsh) {
    auto h_scaling = std::min(0.7, 0.9 * std::exp(epsh / maxerr, (1.0 / (p - 1.0))));
    return choose_osc_stepsize(omega_fun, x0, h_scaling, epsh);
  } else {
    return h;
  }
}


}

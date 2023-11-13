#ifndef INCLUDE_RICATTI_EVOLVE_HPP
#define INCLUDE_RICATTI_EVOLVE_HPP

#include <ricatti/utils.hpp>
#include <complex>
#include <type_traits>

namespace ricatti {
template <typename SolverInfo, typename Scalar, typename Y0>
auto osc_evolve(SolverInfo&& info, Scalar x0, Scalar x1, Scalar h, Y0 y0, Scalar epsres, Scalar epsh) {
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
 * @brief Solves the differential equation y'' + 2gy' + w^2y = 0 over a given interval.
 *
 * This function solves the differential equation on the interval (xi, xf), starting from the
 * initial conditions y(xi) = yi and y'(xi) = dyi. It keeps the residual of the ODE below eps,
 * and returns an interpolated solution (dense output) at the points specified in xeval.
 *
 * @tparam SolverInfo Type of the solver info object containing differentiation matrices, etc.
 * @tparam Scalar Numeric scalar type, typically float or double.
 * @tparam Vec Type of the vector for dense output values, should match Scalar type.
 *
 * @param[in] info SolverInfo object containing necessary information for the solver.
 * @param[in] xi Starting value of the independent variable.
 * @param[in] xf Ending value of the independent variable.
 * @param[in] yi Initial value of the dependent variable at xi.
 * @param[in] dyi Initial derivative of the dependent variable at xi.
 * @param[in] eps Relative tolerance for the local error of both Riccati and Chebyshev type steps.
 * @param[in] epsh Relative tolerance for choosing the stepsize of Riccati steps.
 * @param[in] xeval List of x-values where the solution is to be interpolated (dense output) and returned.
 * @param[in] hard_stop If true, forces the solver to have a potentially smaller last stepsize to stop exactly at xf.
 * @param[in] warn If true, displays warnings during the run; otherwise, warnings are silenced.
 *
 * @return Tuple containing:
 *         - Vector of x-values at internal steps of the solver (xs).
 *         - Vector of y-values of the dependent variable at internal steps (ys).
 *         - Vector of derivatives of the dependent variable at internal steps (dys).
 *         - Vector indicating the success of each step (1 for success, 0 for failure).
 *         - Vector of complex phases accumulated during each successful Riccati step (phases).
 *         - Vector indicating the types of successful steps taken (1 for Riccati, 0 for Chebyshev).
 *         - Vector of interpolated values of the solution at xeval (yeval).
 */
template <typename SolverInfo, typename Scalar, typename Vec>
auto solve(SolverInfo&& info, Scalar xi, Scalar xf, std::complex<Scalar> yi, std::complex<Scalar> dyi,
           Scalar eps, Scalar epsh, Vec&& xeval, bool hard_stop = false, bool warn = false) {
    using Complex = std::complex<Scalar>;
    using VectorXc = Eigen::VectorXcd;
    using MatrixXc = Eigen::MatrixXcd;
    using VectorXs = Eigen::VectorXd;

    // Variables initialization
    VectorXs xs;
    VectorXc ys, dys;
    std::vector<int> successes, steptypes;
    VectorXc yeval;
    VectorXc phases;
    Scalar hslo, hosc, h, wnext, dwnext, hslo_ini, hosc_ini;
    int intdir, success, steptype;
    Complex y, dy, yprev, dyprev;

    // Dense output setup
    bool denseout = !xeval.isZero(0);
    if (denseout) {
        yeval.resize(xeval.size());
    }

    // Check if stepsize sign is consistent with direction of integration
    Scalar hi = info.h0;
    if ((xf - xi) * hi < 0) {
        if (warn) {
            std::cerr << "Direction of integration does not match stepsize sign, adjusting it." << std::endl;
        }
        hi *= -1;
    }

    // Determine direction
    intdir = (hi > 0) ? 1 : -1;

    // Initialization
    xs.push_back(xi);
    ys.push_back(yi);
    dys.push_back(dyi);
    steptypes.push_back(0);
    successes.push_back(1);
    y = yi;
    dy = dyi;
    yprev = y;
    dyprev = dy;

    // Initial stepsize determination
    hslo_ini = intdir * std::min(1e8, std::abs(1 / info.w(xi + hi / 2)));
    hosc_ini = intdir * std::min(1e8, std::abs(info.w(xi + hi / 2) / info.dw(xi + hi / 2)));
    hosc = choose_osc_stepsize(info, xi, hosc_ini, epsh);
    hslo = choose_nonosc_stepsize(info, xi, hslo_ini);
    Scalar xcurrent = xi;

    // Main loop
    while (std::abs(xcurrent - xf) > 1e-8 && intdir * xcurrent < intdir * xf) {
        // Oscillatory or non-oscillatory decision
        success = 0;
        if (intdir * hosc > intdir * hslo * 5) {
            // Oscillatory step
            std::tie(y, dy, success, h) = osc_step(info, xcurrent, hosc, yprev, dyprev, eps);
            steptype = 1;
        } else {
            // Non-oscillatory step
            std::tie(y, dy, success, h) = nonosc_step(info, xcurrent, hslo, yprev, dyprev, eps);
            steptype = 0;
        }

        if (success == 0) {
            throw std::runtime_error("Stepsize became too small, solution didn't converge");
        }

        // Log step
        xs.push_back(xcurrent + h);
        ys.push_back(y);
        dys.push_back(dy);
        steptypes.push_back(steptype);
        successes.push_back(success);

        // Dense output check
        if (denseout) {
            // Implement dense output computation here
        }

        // Advance independent variable and compute next stepsizes
        xcurrent += h;
        hslo_ini = intdir * std::min(1e8, std::abs(1 / info.w(xcurrent)));
        hosc_ini = intdir * std::min(1e8, std::abs(info.w(xcurrent) / info.dw(xcurrent)));
        hosc = choose_osc_stepsize(info, xcurrent, hosc_ini, epsh);
        hslo = choose_nonosc_stepsize(info, xcurrent, hslo_ini);
        yprev = y;
        dyprev = dy;
    }

    return std::make_tuple(xs, ys, dys, successes, phases, steptypes, yeval);
}


}

#endif

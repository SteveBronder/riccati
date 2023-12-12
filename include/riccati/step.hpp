#ifndef INCLUDE_riccati_STEP_HPP
#define INCLUDE_riccati_STEP_HPP

#include <riccati/chebyshev.hpp>
#include <Eigen/Dense>
#include <complex>
#include <cmath>
#include <tuple>

namespace riccati {

template <typename SolverInfo, typename Scalar>
inline auto nonosc_step(SolverInfo &&info, Scalar x0, Scalar h,
                        std::complex<Scalar> y0, std::complex<Scalar> dy0,
                        Scalar epsres = Scalar(1e-12)) {
  using complex_t = std::complex<Scalar>;

  Scalar maxerr = 10 * epsres;
  auto N = info.nini_;
  auto Nmax = info.nmax_;

  auto cheby = spectral_chebyshev(info, x0, h, y0, dy0, 0);
  auto &&yprev = std::get<0>(cheby);
  auto &&dyprev = std::get<1>(cheby);
  auto &&xprev = std::get<2>(cheby);
  while (maxerr > epsres) {
    N *= 2;
    if (N > Nmax) {
      return std::make_tuple(complex_t(0.0, 0.0), complex_t(0.0, 0.0), maxerr,
                             0, yprev, dyprev);
    }
    cheby = spectral_chebyshev(info, x0, h, y0, dy0,
                               static_cast<int>(std::log2(N / info.nini_)));
    auto &&y = std::get<0>(cheby);
    auto &&dy = std::get<1>(cheby);
    auto &&x = std::get<2>(cheby);
    maxerr = std::abs((yprev(0, 0) - y(0, 0)) / y(0, 0));
    if (std::isnan(maxerr)) {
      maxerr = std::numeric_limits<Scalar>::infinity();
    }
    // This might be wrong?
    yprev = y;
    dyprev = dy;
    xprev = x;
  }
  return std::make_tuple(yprev(0, 0), dyprev(0, 0), maxerr, 1, yprev, dyprev);
}

template <typename SolverInfo, typename Scalar>
inline auto osc_step(SolverInfo &&info, Scalar x0, Scalar h,
                     std::complex<Scalar> y0, std::complex<Scalar> dy0,
                     Scalar epsres = Scalar(1e-12), bool plotting = false,
                     int k = 0) {
  using complex_t = std::complex<Scalar>;
  using vectorc_t = vector_t<complex_t>;

  int success = 1;
  auto &&omega_s = info.omega_n_;
  auto &&gamma_s = info.gamma_n_;

  auto &&Dn = info.Dn_;
  vectorc_t y = complex_t(0.0, 1.0) * omega_s;
  auto delta = [&](const auto &r, const auto &y) {
    return (-r.array() / (2.0 * (y.array() + gamma_s.array()))).matrix().eval();
  };
  auto R = [&](const auto &d) {
    return 2.0 / h * (Dn * d) + d.array().square().matrix();
  };
  auto Ry = (complex_t(0.0, 1.0) * 2.0
             * (1.0 / h * (Dn * omega_s) + gamma_s.cwiseProduct(omega_s)))
                .eval();
  Scalar maxerr = Ry.array().abs().maxCoeff();

  vectorc_t deltay;
  Scalar prev_err = std::numeric_limits<Scalar>::infinity();
  while (maxerr > epsres) {
    deltay = delta(Ry, y);
    y += deltay;
    Ry = R(deltay);
    maxerr = Ry.array().abs().maxCoeff();
    if (maxerr >= prev_err) {
      success = 0;
      break;
    }
    prev_err = maxerr;
  }
  vectorc_t du1 = y;
  if (info.denseout_) {
    vectorc_t u1 = h / 2.0 * (info.integration_matrix_ * du1);
    vectorc_t f1 = (u1).array().exp();
    auto f2 = f1.conjugate().eval();
    auto du2 = du1.conjugate().eval();
    auto ap_top = (dy0 - y0 * du2(du2.size() - 1));
    auto ap_bottom = (du1(du1.size() - 1) - du2(du2.size() - 1));
    auto ap = ap_top
               / ap_bottom;
    auto am = (dy0 - y0 * du1(du1.size() - 1))
              / (du2(du2.size() - 1) - du1(du1.size() - 1));
    auto y1 = (ap * f1 + am * f2).eval();
    auto dy1 = (ap * du1.cwiseProduct(f1) + am * du2.cwiseProduct(f2)).eval();
    Scalar phase = std::imag(f1(0));
    info.un_ = u1;
    info.a_ = std::make_pair(ap, am);
    return std::make_tuple(y1(0), dy1(0), maxerr, success, phase);
  } else {
    complex_t f1 = std::exp(h / 2.0 * (info.quadwts_.dot(du1)));
    auto f2 = std::conj(f1);
    auto du2 = du1.conjugate().eval();
    auto ap_top = (dy0 - y0 * du2(du2.size() - 1));
    auto ap_bottom = (du1(du1.size() - 1) - du2(du2.size() - 1));
    auto ap = ap_top
               / ap_bottom;
    auto am = (dy0 - y0 * du1(du1.size() - 1))
              / (du2(du2.size() - 1) - du1(du1.size() - 1));
    auto y1 = (ap * f1 + am * f2);
    auto dy1 = (ap * du1 * f1 + am * du2 * f2).eval();
    Scalar phase = std::imag(f1);
    return std::make_tuple(y1, dy1(0), maxerr, success, phase);
  }
}

}  // namespace riccati

#endif

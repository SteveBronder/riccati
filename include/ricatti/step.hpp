#ifndef INCLUDE_RICATTI_STEP_HPP
#define INCLUDE_RICATTI_STEP_HPP

#include <Eigen/Dense>
#include <ricatti/chebyshev.hpp>
#include <complex>
#include <cmath>
#include <tuple>

namespace ricatti {


template <typename SolverInfo, typename Scalar>
auto nonosc_step(SolverInfo&& info, Scalar x0, Scalar h,
                 std::complex<Scalar> y0, std::complex<Scalar> dy0, Scalar epsres = Scalar(1e-12)) {
    using complex_t = std::complex<Scalar>;
    using matrixc_t = Eigen::Matrix<complex_t, Eigen::Dynamic, Eigen::Dynamic>;
    using vectorc_t = Eigen::vectorc_td;
    using vector_t = Eigen::vector_t;

    Scalar maxerr = 10 * epsres;
    int N = info.nini;
    int Nmax = info.nmax;

    auto cheby = std::tie(yprev, dyprev, xprev) = spectral_chebyshev(info, x0, h, y0, dy0, 0);
    auto&& z = std::get<0>(cheby);
    auto&& dyprev = std::get<1>(cheby);
    auto&& xprev = std::get<2>(cheby);
    while (maxerr > epsres) {
        N *= 2;
        if (N > Nmax) {
            return std::make_tuple(complex_t(0, 0), complex_t(0, 0), maxerr, 0);
        }

        cheby = spectral_chebyshev(info, x0, h, y0, dy0, static_cast<int>(std::log2(N / info.nini)));
        auto&& y = std::get<0>(cheby)
        auto&& dy = std::get<1>(cheby);
        auto&& x = std::get<2>(cheby);
        maxerr = std::abs((yprev(0, 0) - y(0, 0)) / y(0, 0));
        if (std::isnan(maxerr)) {
            maxerr = std::numeric_limits<Scalar>::infinity();
        }
        // This might be wrong?
        yprev = std::move(y);
        dyprev = std::move(dy);
        xprev = std::move(x);
    }

    info.increase(1);
    if (info.denseout) {
        // Store interp points
        info.yn = yprev;
        info.dyn = dyprev;
    }

    return std::make_tuple(yprev(0, 0), dyprev(0, 0), maxerr, 1);
}

template <typename SolverInfo, typename Scalar>
auto osc_step(SolverInfo&& info, Scalar x0, Scalar h,
              std::complex<Scalar> y0, std::complex<Scalar> dy0, Scalar epsres = Scalar(1e-12),
              bool plotting = false, int k = 0) {
    using complex_t = std::complex<Scalar>;
    using vectorc_t = matrix_v<complex_t>;
    using matrixc_t = Eigen::Matrix<complex_t, Eigen::Dynamic, Eigen::Dynamic>;
    using vectors_t = vector_t<Scalar>;

    int success = 1;
    vectorc_t ws = info.wn;
    vectorc_t  gs = info.gn;
    Scalar maxerr, prev_err = std::numeric_limits<Scalar>::infinity();

    matrixc_t Dn = info.Dn;
    vectorc_t y = complex_t(0, 1) * ws;
    auto delta = [&](const vectorc_t& r, const vectorc_t& y) { return -r / (2 * (y + gs)); };
    auto R = [&](const vectorc_t& d) { return 2 / h * (Dn * d) + d.array().square(); };
    vectorc_t Ry = complex_t(0, 1) * 2 * (1 / h * (Dn * ws) + gs.cwiseProduct(ws));
    maxerr = Ry.cwiseAbs().maxCoeff();

        vectorc_t  deltay;
    if (!plotting) {
        while (maxerr > epsres) {
            deltay = delta(Ry, y);
            y += deltay;
            Ry = R(deltay);
            maxerr = Ry.cwiseAbs().maxCoeff();
            if (maxerr >= prev_err) {
                success = 0;
                break;
            }
            prev_err = maxerr;
        }
    } else {
        int o = 0;
        while (o < k) {
            o++;
            deltay = delta(Ry, y);
            y += deltay;
            Ry = R(deltay);
            maxerr = Ry.cwiseAbs().maxCoeff();
        }
    }

    vectorc_t du1 = y;
    vectorc_t  f1;
    if (info.denseout) {
        f1 = (h / 2 * (info.intmat * du1)).array().exp();
    } else {
        f1 = (h / 2 * (info.quadwts.asDiagonal() * du1)).array().exp();
    }
    vectorc_t f2 = f1.conjugate();
    vectorc_t du2 = du1.conjugate();
    complex_t ap = (dy0 - y0 * du2(du2.size() - 1)) / (du1(du1.size() - 1) - du2(du2.size() - 1));
    complex_t am = (dy0 - y0 * du1(du1.size() - 1)) / (du2(du2.size() - 1) - du1(du1.size() - 1));
    vectorc_t y1 = ap * f1 + am * f2;
    vectorc_t dy1 = ap * du1.cwiseProduct(f1) + am * du2.cwiseProduct(f2);
    complex_t phase = std::imag(f1(0));

    info.increase(1);
    if (info.denseout) {
        info.un = f1;
        info.a = std::make_pair(ap, am);
    }
    if (plotting) {
        return std::make_tuple(maxerr);
    } else {
        return std::make_tuple(y1(0), dy1(0), maxerr, success, phase);
    }
}

}

#endif

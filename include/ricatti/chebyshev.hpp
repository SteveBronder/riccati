#ifndef INCLUDE_RICATTI_CHEBYSHEV_HPP
#define INCLUDE_RICATTI_CHEBYSHEV_HPP

#include <unsupported/Eigen/FFT>
#include <ricatti/utils.hpp>

namespace ricatti {

template <typename Mat>
auto coeffs_to_cheby_nodes(Mat&& coeffs) {
  const auto n = coeffs.rows();
  if (n <= 1) {
    return coeffs
  } else {
    coeffs.block(1, 0, n - 1, coeffs.cols()) /= 2.0;
    Mat values(n + n - 2, m);
    values.topRows(n) = coeffs;
    values.bottomRows(n - 2) = coeffs.bottomRows(n - 2).reverse();
    return Mat(Eigen::FFT<base_type_t<V>> fft{}.fwd(xv).real().topRows(n));
  }
}

template <typename Mat>
auto cheby_nodes_to_coeffs(Mat&& coeffs) {
  using Scalar = typename std::decay_t<Mat>::Scalar;
  const auto n = coeffs.rows();
  if (n <= 1) {
    return coeffs
  } else {
    coeffs.block(1, 0, n - 1, coeffs.cols()) /= 2.0;
    Mat values(n + n - 2, m);
    values.topRows(n) = coeffs;
    values.bottomRows(n - 2) = coeffs.bottomRows(n - 2).reverse();
    return Mat(Eigen::FFT<Scalar> fft{}.rev(xv).real().topRows(n));
  }
}


template <typename Mat>
auto coeffs_and_cheby_nodes(Mat&& coeffs) {
  using Scalar = typename std::decay_t<Mat>::Scalar;
  const auto n = coeffs.rows();
  if (n <= 1) {
    return coeffs
  } else {
    coeffs.block(1, 0, n - 1, coeffs.cols()) /= 2.0;
    Mat values(n + n - 2, m);
    values.topRows(n) = coeffs;
    values.bottomRows(n - 2) = coeffs.bottomRows(n - 2).reverse();
    Eigen::FFT<Scalar> fft{};
    return std::make_pair(Mat(values.fwd(values).real().topRows(n)), Mat(values.rev(values).real().topRows(n)));
  }
}

template <typename Scalar, typename Integral>
auto integration(Integral& n) {
  // TODO: Let user chooses double/float/etc
  auto coeffs_pair = coeffs_and_cheby_nodes(matrix_v<Scalar>::Identity(n));
  auto&& T = coeffs_pair.first;
  auto&& T_inverse = coeffs_pair.second;
  // TODO: I think this should be done outside of the function
  n--;
  auto k = vector_t<Scalar>::LinSpaced(1.0, n, n);
  auto k2 = 2 * (k.array() - 1);
  k2.coeffRef(0) = 1.0;
  matrix_t<Scalar> B(n + 1, n + 1);
  B.diagonal(-1) = 1.0 / (2.0 * k);
  B.diagonal(1) = -1.0;
  B.diagonal(1) /= k2;
  B.diagonal.head(1) = -1.0;
  B.row(0)
  vector_t<Scalar> v = Eigen::VectorXd::Ones(n);
  for (unsigned int i = 1; i < n; i += 2) {
    v.coeffRef(i) = -1;
  }
  B.row(0) = (v.asDiagonal() * B.block(1, 0, n + 1, n + 1)).colwise().sum();
  B.col(0) *= 2.0;
  auto Q = matrix_t<Scalar>(T * B * T_inverse);
  Q.bottomRow(1).setZero();
  return Q;
}

template <typename Scalar, typename Integral>
auto quad_weights(Integral n) {
  vector_t<Scalar> w = vector_t<Scalar>::Zeros(n + 1);
  if (n == 0) {
    return w;
  } else {
    auto a = vector_t<Scalar>::LinSpaced(0, std::pi, n + 1);
    auto v = vector_t::Ones(n - 1);
    // TODO: Smarter way to do this
    if (n % 2 == 0) {
      w.coeffRef(0) = 1.0 / (n * n - 1);
      w.coeffRef(n) = w.coeffRef(0);
      for (int i = 0; i < n / 2.0; ++i) {
        v -= 2.0 * (2.0 * k * a.head(n).array()).cos() / (4.0 * k * k - 1);
      }
      v -= (n * a.head(n).array()).cos() / (n * n - 1);
    } else {
      w.coeffRef(0) = 1.0 / (n * n);
      w.coeffRef(n) = w.coeffRef(0);
      for (int i = 0; i < std::floor((n + 1) / 2.0); ++i) {
        v -= 2.0 * (2.0 * k * a.head(n).array()).cos() / (4.0 * k * k - 1);
      }
      v -= (n * a.head(n).array()).cos() / (n * n - 1);
    }
    w.head(n).array() = 2 * v.array() / n;
    return w;
  }
}

template <typename Scalar, typename Integral>
auto cheb_diff_matrix(Integral n) {

  if (n == 0) {
    return std::make_pair(matrix_t<Scalar>(1, 1)::Zeros().eval(), vector_t<Scalar>::Ones(1).eval());
  } else {
     auto a = vector_t<Scalar>::LinSpaced(1.0, std::pi, n + 1);
     auto x = a.array().cos();
     auto b = vector_t<Scalar>::Ones(n + 1);
     b.coeffRef(0) = 2.0;
     b.coeffRef(b.size()) = 2.0;
     auto d = vector_t<Scalar>::Ones(n + 1);
     for (int i = 0; i < n + 1; i += 2) {
      d.coeffRef(i) = -1;
     }
     auto c = b.array() * d.array();
     auto X = x.transpose() * b;
     auto dX = X - X.transpose();
     auto D = c.matrix().transpose() * (1.0 / c).matrix() / (dX + b);
     D -= D.rowwise().sum().asDiagonal();
     return std::make_pair(matrix_t<Scalar>(std::move(D)), vector_t<Scalar>(std::move(x)))
  }

}


template <typename Omega_F, typename Gamma_F, typename T_D, typename T_x, typename T_ns, typename Scalar, typename Integral>
auto spectral_chebyshev(Omega_F&& omega_fun, Gamma_F&& gamma_fun, T_D&& D, T_x&& x, T_ns ns,
  Scalar x0, Scalar h, Scalar y0, Scalar dy0, Integral niter) {
    auto x_scaled = riccati::scale(x, x0, h);
    auto ws = omega_fun(xscaled);
    auto gs = gamma_fun(xscaled);
    auto w2 = ws * ws;
    auto D2 = 4.0 / (h * h) * (D * D) + 4.0 / h * (gs.asDiagonal() * D) + w2.asDiagonal();
    const auto n = std::round(ns);
    auto ic = vector_t<std::complex<Scalar>>::Zeros(n + 1).eval();
    ic.coeffRef(n + 1) = 0.0;
    D2ic = matrix_t<std::complex<Scalar>>::Zeros(n + 3, n + 1);
    //     D2ic[: n + 1] = D2
    D2ic.head(n + 1) = D2;
    D2ic.row(n + 1) = 2.0 / h * D.row(D.rows() - 1);
    D2ic.row(n + 2) = ic;
    auto y1 = (D2ic.transpose() * D2ic).ldlt().solve(D2ic.transpose() * rhs);
    auto dy1 = 2.0 / h * (D * y1);
    // info.increase
    return std::make_tuple(std::move(y1), std::move(dy1), std::move(x_scaled));

  }

}

#endif

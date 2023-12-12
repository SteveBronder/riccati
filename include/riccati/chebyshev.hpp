#ifndef INCLUDE_riccati_CHEBYSHEV_HPP
#define INCLUDE_riccati_CHEBYSHEV_HPP

#include <riccati/utils.hpp>
#include <unsupported/Eigen/FFT>

namespace riccati {

namespace internal {
template <bool Fwd, typename T>
inline auto fft(T&& x) {
  using Scalar = typename std::decay_t<T>::Scalar;
  Eigen::FFT<Scalar> fft;
  using T_t = std::decay_t<T>;
  typename T_t::PlainObject res(x.rows(), x.cols());
  for (Eigen::Index j = 0; j < x.cols(); ++j) {
    if (Fwd) {
      res.col(j) = fft.fwd(Eigen::Matrix<std::complex<Scalar>, -1, 1>(x.col(j)))
                       .real();
    } else {
      res.col(j) = fft.inv(Eigen::Matrix<std::complex<Scalar>, -1, 1>(x.col(j)))
                       .real();
    }
  }
  return res;
}
}  // namespace internal
template <typename Mat>
inline auto coeffs_to_cheby_nodes(Mat&& input_id) {
  using Scalar = typename std::decay_t<Mat>::Scalar;
  const auto n = input_id.rows();
  using Mat_t = matrix_t<Scalar>;
  if (n <= 1) {
    return Mat_t(input_id);
  } else {
    Mat_t fwd_values(n + n - 2, input_id.cols());
    fwd_values.topRows(n) = input_id;
    fwd_values.block(1, 0, n - 2, n) /= 2.0;
    fwd_values.bottomRows(n - 1)
        = fwd_values.topRows(n - 1).rowwise().reverse();
    return Mat_t(internal::fft<true>(fwd_values).topRows(n).eval());
  }
}

template <typename Mat>
inline auto cheby_nodes_to_coeffs(Mat&& input_id) {
  using Scalar = typename std::decay_t<Mat>::Scalar;
  using Mat_t = matrix_t<Scalar>;
  const auto n = input_id.rows();
  if (n <= 1) {
    return Mat_t(Mat_t::Zero(input_id.rows(), input_id.cols()));
  } else {
    Mat_t rev_values(n + n - 2, input_id.cols());
    rev_values.topRows(n) = input_id;
    rev_values.bottomRows(n - 1)
        = rev_values.topRows(n - 1).rowwise().reverse();
    auto rev_ret = Mat_t(internal::fft<false>(rev_values).topRows(n)).eval();
    rev_ret.block(1, 0, n - 2, n).array() *= 2.0;
    return rev_ret;
  }
}

template <typename Mat>
inline auto coeffs_and_cheby_nodes(Mat&& input_id) {
  using Scalar = typename std::decay_t<Mat>::Scalar;
  using Mat_t = matrix_t<Scalar>;
  const auto n = input_id.rows();
  if (n <= 1) {
    return std::make_pair(Mat_t(input_id),
                          Mat_t(Mat_t::Zero(input_id.rows(), input_id.cols())));
  } else {
    Mat_t fwd_values(n + n - 2, input_id.cols());
    fwd_values.topRows(n) = input_id;
    fwd_values.block(1, 0, n - 2, n) /= 2.0;
    fwd_values.bottomRows(n - 1)
        = fwd_values.topRows(n - 1).rowwise().reverse();
    auto fwd_val = Mat_t(internal::fft<true>(fwd_values).topRows(n).eval());
    Mat_t rev_values(n + n - 2, input_id.cols());
    rev_values.topRows(n) = input_id;
    rev_values.bottomRows(n - 1)
        = rev_values.topRows(n - 1).rowwise().reverse();
    auto rev_ret = Mat_t(internal::fft<false>(rev_values).topRows(n)).eval();
    rev_ret.block(1, 0, n - 2, n).array() *= 2.0;
    return std::make_pair(fwd_val, rev_ret);
  }
}

template <typename Scalar, typename Integral>
inline auto integration_matrix(Integral n) {
  auto ident = matrix_t<Scalar>::Identity(n, n).eval();
  auto coeffs_pair = coeffs_and_cheby_nodes(ident);
  auto&& T = coeffs_pair.first;
  auto&& T_inverse = coeffs_pair.second;
  n--;
  auto k = vector_t<Scalar>::LinSpaced(n, 1.0, n).eval();
  auto k2 = eval(2 * (k.array() - 1));
  k2.coeffRef(0) = 1.0;
  // B = np.diag(1 / (2 * k), -1) - np.diag(1 / k2, 1)
  matrix_t<Scalar> B = matrix_t<Scalar>::Zero(n + 1, n + 1);
  B.diagonal(-1).array() = 1.0 / (2.0 * k).array();
  B.diagonal(1).array() = -1.0 / k2.array();
  vector_t<Scalar> v = Eigen::VectorXd::Ones(n);
  for (Integral i = 1; i < n; i += 2) {
    v.coeffRef(i) = -1;
  }
  auto tmp = (v.asDiagonal() * B.block(1, 0, n, n + 1)).eval();
  B.row(0) = (tmp).colwise().sum();
  B.col(0) *= 2.0;
  auto Q = matrix_t<Scalar>(T * B * T_inverse);
  Q.bottomRows(1).setZero();
  return Q;
}

template <typename Scalar, typename Integral>
inline auto quad_weights(Integral n) {
  vector_t<Scalar> w = vector_t<Scalar>::Zero(n + 1);
  if (n == 0) {
    return w;
  } else {
    auto a = vector_t<Scalar>::LinSpaced(n + 1, 0, pi<Scalar>()).eval();
    auto v = vector_t<Scalar>::Ones(n - 1).eval();
    // TODO: Smarter way to do this
    if (n % 2 == 0) {
      w.coeffRef(0) = 1.0 / static_cast<Scalar>(n * n - 1);
      w.coeffRef(n) = w.coeff(0);
      for (Integral i = 1; i < n / 2; ++i) {
        v.array() -= 2.0 * (2.0 * i * a.segment(1, n - 1).array()).cos()
                     / (4.0 * i * i - 1);
      }
      v.array() -= (n * a.segment(1, n - 1).array()).cos() / (n * n - 1);
    } else {
      w.coeffRef(0) = 1.0 / static_cast<Scalar>(n * n);
      w.coeffRef(n) = w.coeff(0);
      for (std::size_t i = 0; i < std::floor((n + 1) / 2.0); ++i) {
        v.array() -= 2.0 * (2.0 * i * a.segment(1, n - 1).array()).cos()
                     / (4.0 * i * i - 1);
      }
      v.array() -= (n * a.segment(1, n - 1).array()).cos() / (n * n - 1);
    }
    w.segment(1, n - 1).array() = (2.0 * v.array()) / n;
    return w;
  }
}

template <typename Scalar, typename Integral>
inline auto chebyshev(Integral n) {
  if (n == 0) {
    return std::make_pair(matrix_t<Scalar>::Zero(1, 1).eval(),
                          vector_t<Scalar>::Ones(1).eval());
  } else {
    auto a = vector_t<Scalar>::LinSpaced(n + 1, 0.0, pi<Scalar>());
    auto b = vector_t<Scalar>::Ones(n + 1).eval();
    b.coeffRef(0) = 2.0;
    b.coeffRef(b.size() - 1) = 2.0;
    auto d = vector_t<Scalar>::Ones(n + 1).eval();
    for (Integral i = 1; i < n + 1; i += 2) {
      d.coeffRef(i) = -1;
    }
    auto x = a.array().cos().eval();
    auto X = (x.matrix() * vector_t<Scalar>::Ones(n + 1).transpose())
                 .matrix()
                 .eval();
    auto dX = (X - X.transpose()).eval();
    auto c = (b.array() * d.array()).eval();

    auto D = ((c.matrix() * (1.0 / c).matrix().transpose()).array()
              / (dX + matrix_t<Scalar>::Identity(n + 1, n + 1)).array())
                 .matrix()
                 .eval();
    D -= D.rowwise().sum().asDiagonal();
    return std::make_pair(matrix_t<Scalar>(D), vector_t<Scalar>(x));
  }
}

template <typename Vec1, typename Vec2>
inline auto interpolate(Vec1&& s, Vec2&& t) {
  const auto r = s.size();
  const auto q = t.size();
  auto V = matrix_t<typename std::decay_t<Vec1>::Scalar>::Ones(r, r).eval();
  auto R = matrix_t<typename std::decay_t<Vec1>::Scalar>::Ones(q, r).eval();
  for (std::size_t i = 1; i < static_cast<std::size_t>(r); ++i) {
    V.col(i).array() = V.col(i - 1).array() * s.array();
    R.col(i).array() = R.col(i - 1).array() * t.array();
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      V.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::MatrixXd L = svd.solve(R.transpose().eval()).transpose();
  return L;
}

template <typename SolverInfo, typename Scalar, typename Integral>
inline auto spectral_chebyshev(SolverInfo&& info, Scalar x0, Scalar h,
                               std::complex<Scalar> y0,
                               std::complex<Scalar> dy0, Integral niter) {
  using complex_t = std::complex<Scalar>;
  using vectorc_t = vector_t<complex_t>;
  auto x_scaled = riccati::scale(info.chebyshev_[niter].second, x0, h).eval();
  auto&& D = info.chebyshev_[niter].first;
  auto ws = info.omega_fun_(x_scaled);
  auto gs = info.gamma_fun_(x_scaled);
  auto w2 = (ws.array() * ws.array()).matrix();
  auto D2 = (4.0 / (h * h) * (D * D) + 4.0 / h * (gs.asDiagonal() * D)).eval();
  D2 += w2.asDiagonal();
  const auto n = std::round(info.ns_[niter]);
  auto D2ic = matrix_t<complex_t>::Zero(n + 3, n + 1).eval();
  D2ic.topRows(n + 1) = D2;
  D2ic.row(n + 1) = 2.0 / h * D.row(D.rows() - 1);
  auto ic = vectorc_t::Zero(n + 1).eval();
  ic.coeffRef(n) = complex_t{1.0, 0.0};
  D2ic.row(n + 2) = ic;
  vectorc_t rhs = vectorc_t::Zero(n + 3);
  rhs.coeffRef(n + 1) = dy0;
  rhs.coeffRef(n + 2) = y0;
  vectorc_t y1 = D2ic.colPivHouseholderQr().solve(rhs);
  auto dy1 = (2.0 / h * (D * y1)).eval();
  return std::make_tuple(std::move(y1), std::move(dy1), std::move(x_scaled));
}

}  // namespace riccati

#endif

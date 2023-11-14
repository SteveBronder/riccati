#ifndef INCLUDE_riccati_UTILS_HPP
#define INCLUDE_riccati_UTILS_HPP

namespace riccati {
template <typename Scalar, typename Vector>
auto scale(Vector &&x, Scalar x0, Scalar h) {
  return x0 + h / 2.0 + h / 2.0 * x;
}

template <typename Scalar>
using matrix_t = Eigen::Matrix<Scalar, -1, -1>;
template <typename Scalar>
using vector_t = Eigen::Matrix<Scalar, -1, 1>;

template <typename Scalar>
using row_vector_t = Eigen::Matrix<Scalar, 1, -1>;

template <typename Scalar>
using array2d_t = Eigen::Matrix<Scalar, -1, -1>;
template <typename Scalar>
using array1d_t = Eigen::Matrix<Scalar, -1, 1>;
template <typename Scalar>
using row_array1d_t = Eigen::Matrix<Scalar, 1, -1>;

template <typename T>
inline constexpr T pi() {
  return static_cast<T>(3.141592653589793238463);
}
}  // namespace riccati

#endif

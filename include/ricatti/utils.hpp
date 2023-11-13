#ifndef INCLUDE_RICATTI_UTILS_HPP
#define INCLUDE_RICATTI_UTILS_HPP

namespace ricatti {
template <typename Scalar, typename Vector>
auto scale(Vector&& x, Scalar x0, Scalar h) {
  return x0 + h / 2.0 + h / 2.0 * x;
}

template <typename Scalar>
using matrix_t = Eigen::Matrix<Scalar, -1, -1>;
template <typename Scalar>
using vector_t = Eigen::Matrix<Scalar, -1, 1>;
template <typename Scalar>
using vector_t = Eigen::Matrix<Scalar, 1, -1>;

template <typename Scalar>
using array2d_t = Eigen::Matrix<Scalar, -1, -1>;
template <typename Scalar>
using array1dc_t = Eigen::Matrix<Scalar, -1, 1>;
template <typename Scalar>
using array1dr_t = Eigen::Matrix<Scalar, 1, -1>;

}


#endif

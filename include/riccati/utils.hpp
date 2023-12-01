#ifndef INCLUDE_riccati_UTILS_HPP
#define INCLUDE_riccati_UTILS_HPP

#include <Eigen/Dense>
#define RICCATI_DEBUG
#ifdef RICCATI_DEBUG
#include <iostream>
#endif
namespace riccati {
template <typename Scalar, typename Vector>
inline auto scale(Vector &&x, Scalar x0, Scalar h) {
  return (x0 + h / 2.0 + h / 2.0 * x.array()).matrix();
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

inline double eval(double x) {
  return x;
}
template <typename T>
inline std::complex<T>& eval(std::complex<T>& x) {
  return x;
}
template <typename T>
inline std::complex<T> eval(std::complex<T>&& x) {
  return std::move(x);
}

template <typename T>
inline auto eval(T&& x) {
  return x.eval();
}

template <typename T, Eigen::Index R, Eigen::Index C>
inline auto eval(Eigen::Matrix<T, R, C>&& x) {
  return std::move(x);
}

template <typename T, Eigen::Index R, Eigen::Index C>
inline  auto& eval(Eigen::Matrix<T, R, C>& x) {
  return x;
}

template <typename T, Eigen::Index R, Eigen::Index C>
inline  const auto& eval(const Eigen::Matrix<T, R, C>& x) {
  return x;
}

template <typename T, Eigen::Index R, Eigen::Index C>
inline auto eval(Eigen::Array<T, R, C>&& x) {
  return std::move(x);
}

template <typename T, Eigen::Index R, Eigen::Index C>
inline auto& eval(Eigen::Array<T, R, C>& x) {
  return x;
}

template <typename T, Eigen::Index R, Eigen::Index C>
inline const auto& eval(const Eigen::Array<T, R, C>& x) {
  return x;
}

template <typename T>
void print_matrix(const char* name, T&& x) {
#ifdef RICCATI_DEBUG
  std::cout << name << "(" << x.rows() << ", " << x.cols() << ")" << std::endl;
  std::cout << x << std::endl;
#endif
}


}  // namespace riccati

#endif

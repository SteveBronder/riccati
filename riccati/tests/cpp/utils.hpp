#ifndef RICCATI_TESTS_CPP_UTILS_HPP
#define RICCATI_TESTS_CPP_UTILS_HPP
#include <boost/math/special_functions/airy.hpp>
#include <type_traits>

namespace riccati {
namespace test {
template <typename T>
using require_not_floating_point
    = std::enable_if_t<!std::is_floating_point<std::decay_t<T>>::value>;

template <typename T>
using require_floating_point
    = std::enable_if_t<std::is_floating_point<std::decay_t<T>>::value>;


namespace internal {
  template <typename T>
  struct is_complex_impl : std::false_type {};
  template <typename T>
  struct is_complex_impl<std::complex<T>> : std::true_type {};
}

template <typename T>
struct is_complex : internal::is_complex_impl<std::decay_t<T>> {};

template <typename T>
using require_floating_point_or_complex
    = std::enable_if_t<
      std::is_floating_point<std::decay_t<T>>::value ||
      is_complex<std::decay_t<T>>::value>;

template <typename T>
using require_not_floating_point_or_complex
    = std::enable_if_t<
      !std::is_floating_point<std::decay_t<T>>::value &&
      !is_complex<std::decay_t<T>>::value>;

template <typename T, require_floating_point_or_complex<T>* = nullptr>
inline auto sin(T x) { return std::sin(x); }

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto sin(T&& x) {
  return x.sin();
}

template <typename T, require_floating_point_or_complex<T>* = nullptr>
inline auto cos(T x) { return std::cos(x); }

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto cos(T&& x) {
  return x.cos();
}

template <typename T, require_floating_point_or_complex<T>* = nullptr>
inline auto sqrt(T x) { return std::sqrt(x); }

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto sqrt(T&& x) {
  return x.sqrt();
}

template <typename T, require_floating_point_or_complex<T>* = nullptr>
inline auto array(T x) { return x; }

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto array(T&& x) {
  return x.array();
}

template <typename T, require_floating_point_or_complex<T>* = nullptr>
inline auto matrix(T x) { return x; }

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto matrix(T&& x) {
  return x.matrix();
}

template <typename T, require_floating_point_or_complex<T>* = nullptr>
inline auto eval(T x) { return x; }

template <typename T, require_not_floating_point_or_complex<T>* = nullptr>
inline auto eval(T&& x) {
  return x.eval();
}

template <typename T, require_floating_point_or_complex<T>* = nullptr>
inline constexpr T zero_like(T x) { return static_cast<T>(0); }

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto zero_like(const T& x) {
  return std::decay_t<typename T::PlainObject>::Zero(x.rows(), x.cols());
}

template <typename T, require_floating_point<T>* = nullptr>
inline auto airy_ai(T x) { return boost::math::airy_ai(x); }

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto airy_ai(T&& x) {
  return x.array()
      .unaryExpr([](auto&& x) { return boost::math::airy_ai(x); })
      .matrix()
      .eval();
}

template <typename T, require_floating_point<T>* = nullptr>
inline auto airy_bi(T x) { return boost::math::airy_bi(x); }

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto airy_bi(T&& x) {
  return x.array()
      .unaryExpr([](auto&& x) { return boost::math::airy_bi(x); })
      .matrix()
      .eval();
}

template <typename T, require_floating_point<T>* = nullptr>
inline auto airy_ai_prime(T x) { return boost::math::airy_ai_prime(x); }

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto airy_ai_prime(T&& x) {
  return x.array()
      .unaryExpr([](auto&& x) { return boost::math::airy_ai_prime(x); })
      .matrix()
      .eval();
}

template <typename T, require_floating_point<T>* = nullptr>
inline auto airy_bi_prime(T x) { return boost::math::airy_bi_prime(x); }

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto airy_bi_prime(T&& x) {
  return x.array()
      .unaryExpr([](auto&& x) { return boost::math::airy_bi_prime(x); })
      .matrix()
      .eval();
}

template <typename T>
inline auto airy_i(T&& xi) {
  return eval(airy_ai(-xi) + std::complex<std::decay_t<T>>(0.0, 1.0) * airy_bi(-xi));
} 

template <typename T>
inline auto airy_i_prime(T&& xi) {
  return eval(-airy_ai_prime(-xi) - std::complex<std::decay_t<T>>(0.0, 1.0) * airy_bi_prime(-xi));
}

}  // namespace test
}  // namespace riccati

#endif

#ifndef RICCATI_TESTS_CPP_UTILS_HPP
#define RICCATI_TESTS_CPP_UTILS_HPP
#include <type_traits>


namespace riccati {
namespace test {
template <typename T>
using require_not_floating_point = std::enable_if_t<!std::is_floating_point<std::decay_t<T>>::value>;

template <typename T>
using require_floating_point = std::enable_if_t<std::is_floating_point<std::decay_t<T>>::value>;

inline auto sin(double x) {
  return std::sin(x);
}

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto sin(T&& x) {
  return x.sin();
}

inline auto cos(double x) {
  return std::cos(x);
}

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto cos(T&& x) {
  return x.cos();
}

inline auto sqrt(double x) {
  return std::sqrt(x);
}



template <typename T, require_not_floating_point<T>* = nullptr>
inline auto sqrt(T&& x) {
  return x.sqrt();
}

inline auto array(double x) {
  return x;
}

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto array(T&& x) {
  return x.array();
}

inline auto matrix(double x) {
  return x;
}

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto matrix(T&& x) {
  return x.matrix();
}

inline auto eval(double x) {
  return x;
}

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto eval(T&& x) {
  return x.eval();
}

inline constexpr auto zero_like(double x) {
  return 0;
}

template <typename T, require_not_floating_point<T>* = nullptr>
inline auto zero_like(const T& x) {
  return std::decay_t<typename T::PlainObject>::Zero(x.rows(), x.cols());
}

}
}

#endif

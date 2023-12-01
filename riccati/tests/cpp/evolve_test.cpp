#include <riccati/evolve.hpp>
#include <tests/cpp/utils.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string>

#include <boost/math/special_functions/airy.hpp>


TEST(riccati, evolve_dense_output) {
  using namespace riccati::test;
  auto omega_fun = [](auto&& x) { return eval(matrix(riccati::test::sqrt(array(x))));};
  auto gamma_fun = [](auto&& x) {
    return zero_like(x);
  };
  auto info = riccati::make_solver<true>(omega_fun, gamma_fun, 0.1, 16, 32, 32, 32);
  auto xi = 1e0;
  auto xf = 1e6;
  auto eps = 1e-12;
  auto epsh = 1e-13;
  using boost::math::airy_ai;
  using boost::math::airy_bi;
  using boost::math::airy_ai_prime;
  using boost::math::airy_bi_prime;
  auto yi = airy_ai(-xi) + std::complex<double>(0, 1) * airy_bi(-xi);
  auto dyi = -airy_ai_prime(-xi) - std::complex<double>(0, 1) * airy_bi_prime(-xi);
  Eigen::Index Neval = 1e2;
  riccati::vector_t<double> x_eval = riccati::vector_t<double>::LinSpaced(Neval, xi, xf);
  auto res = riccati::solve(info, xi, xf, yi, dyi, eps, epsh, x_eval);
}


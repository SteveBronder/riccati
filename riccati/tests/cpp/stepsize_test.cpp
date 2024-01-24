#include <riccati/solver.hpp>
#include <riccati/stepsize.hpp>
#include <tests/cpp/utils.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string>

#include <boost/math/special_functions/airy.hpp>

TEST(riccati, osc_stepsize_dense_output) {
  using namespace riccati::test;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::test::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<true, double>(omega_fun, gamma_fun, 16, 32,
                                                 32, 32);
  auto xi = 1e2;
  auto epsh = 1e-13;
  auto hi = 2.0 * xi;
  hi = riccati::choose_osc_stepsize(info, xi, hi, epsh);
  EXPECT_EQ(hi, 200.0);
}

TEST(riccati, osc_stepsize_nondense_output) {
  using namespace riccati::test;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::test::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<false, double>(omega_fun, gamma_fun, 16, 32,
                                                  32, 32);
  auto xi = 1e2;
  auto epsh = 1e-13;
  auto hi = 2.0 * xi;
  hi = riccati::choose_osc_stepsize(info, xi, hi, epsh);
  EXPECT_EQ(hi, 200.0);
}

TEST(riccati, nonosc_stepsize_dense_output) {
  using namespace riccati::test;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::test::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<true, double>(omega_fun, gamma_fun, 16, 32,
                                                 32, 32);
  auto xi = 1e0;
  auto epsh = 2e-1;
  auto hi = 1.0 / omega_fun(xi);
  hi = riccati::choose_nonosc_stepsize(info, xi, hi, epsh);
  EXPECT_EQ(hi, 0.5);
}

TEST(riccati, nonosc_stepsize_nondense_output) {
  using namespace riccati::test;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::test::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<false, double>(omega_fun, gamma_fun, 16, 32,
                                                  32, 32);
  auto xi = 1e0;
  auto epsh = 2e-1;
  auto hi = 1.0 / omega_fun(xi);
  hi = riccati::choose_nonosc_stepsize(info, xi, hi, epsh);
  EXPECT_EQ(hi, 0.5);
}

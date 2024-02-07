#include <riccati/solver.hpp>
#include <riccati/step.hpp>
#include <tests/cpp/utils.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string>

TEST(riccati, osc_step_test) {
  using namespace riccati::test;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::test::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<true, double>(omega_fun, gamma_fun, 16, 32,
                                                 32, 32);
  auto x0 = 10.0;
  auto h = 20.0;
  auto eps = 1e-12;
  auto xscaled = (x0 + h / 2.0 + h / 2.0 * info.xn_.array()).matrix().eval();
  auto omega_n = info.omega_fun_(xscaled).eval();
  auto gamma_n = info.gamma_fun_(xscaled).eval();
  auto y0 = airy_ai(-x0);
  auto dy0 = -airy_ai_prime(-x0);
  auto res = riccati::osc_step(info, omega_n, gamma_n, x0, h, y0, dy0, eps);
  auto y_ana = airy_ai(-(x0 + h));
  auto dy_ana = -airy_ai_prime(-(x0 + h));
  auto y_err = std::abs((std::get<1>(res) - y_ana) / y_ana);
  auto dy_err = std::abs((std::get<2>(res) - dy_ana) / dy_ana);
  EXPECT_NEAR(y_err, 0, 1e-10);
  EXPECT_NEAR(dy_err, 0, 1e-10);
}

TEST(riccati, nonosc_step_test) {
  using namespace riccati::test;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::test::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<true, double>(omega_fun, gamma_fun, 16, 32,
                                                 32, 32);
  auto xi = 1e0;
  auto h = 0.5;
  auto yi = airy_bi(-xi);
  auto dyi = -airy_bi_prime(-xi);
  auto eps = 1e-12;
  auto* allocator = new riccati::arena_alloc{};
  riccati::arena_allocator<double, riccati::arena_alloc> allocator(allocator);
  auto res = riccati::nonosc_step(info, xi, h, yi, dyi, eps);
  auto y_ana = airy_bi(-xi - h);
  auto dy_ana = -airy_bi_prime(-xi - h);
  auto y_err = std::abs((std::get<1>(res) - y_ana) / y_ana);
  auto dy_err = std::abs((std::get<2>(res) - dy_ana) / dy_ana);
  EXPECT_NEAR(y_err, 0, 1e-10);
  EXPECT_NEAR(dy_err, 0, 1e-10);
  delete allocator;
}

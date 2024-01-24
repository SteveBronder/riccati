#include <riccati/solver.hpp>
#include <tests/cpp/utils.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string>


TEST(riccati, solver_make_solver_nondense) {
  auto omega_f = [](auto& x) {
    return riccati::test::sqrt(x);
  };
  auto gamma_f = [](auto& x) {
    return zero_like(x);
  };
  auto info = riccati::make_solver<false, double>(omega_f, gamma_f, 16UL, 32UL,
16UL, 32UL);
}

TEST(riccati, solver_make_solver_dense) {
  auto omega_fun = [](auto&& x) { return x.sqrt().eval();};
  auto gamma_fun = [](auto&& x) {
    using inp_t = std::decay_t<decltype(x)>;
    using Scalar = typename inp_t::Scalar;
    return Eigen::Matrix<std::complex<Scalar>, -1, 1>::Zero(x.size());
  };
  auto info = riccati::make_solver<true, double>(omega_fun, gamma_fun, 16, 32,
32, 32);

}



#include <riccati/evolve.hpp>
#include <riccati/solver.hpp>
#include <tests/cpp/utils.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string>

TEST_F(Riccati, osc_evolve_dense_output) {
  using namespace riccati::test;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::test::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<true, double>(omega_fun, gamma_fun, 16, 32,
                                                 32, 32);
  auto xi = 1e2;
  auto xf = 1e6;
  auto eps = 1e-12;
  auto epsh = 1e-13;
  auto yi = airy_i(xi);
  auto dyi = airy_i_prime(xi);
  Eigen::Index Neval = 1e3;
  riccati::vector_t<double> x_eval
      = riccati::vector_t<double>::LinSpaced(Neval, xi, xf);
  auto ytrue = airy_i(x_eval.array()).matrix().eval();
  auto hi = 2.0 * xi;
  hi = std::get<0>(choose_osc_stepsize(info, xi, hi, epsh, allocator));
  bool x_validated = false;
  while (xi < xf) {
    auto res
        = riccati::osc_evolve(info, xi, xf, yi, dyi, eps, epsh, hi, x_eval, allocator);
    if (!std::get<0>(res)) {
      break;
    } else {
      xi = std::get<1>(res);
      hi = std::get<2>(res);
      yi = std::get<1>(std::get<3>(res));
      dyi = std::get<2>(std::get<3>(res));
    }
    auto airy_true = airy_i(xi);
    auto airy_est = yi;
    auto err = std::abs((airy_true - airy_est) / airy_true);
    EXPECT_LE(err, 3e-7);
    auto start_y = std::get<5>(res);
    auto size_y = std::get<6>(res);
    if (size_y > 0) {
      x_validated = true;
      auto y_true_slice = ytrue.segment(start_y, size_y);
      auto y_err
          = ((std::get<4>(res) - y_true_slice).array() / y_true_slice.array())
                .abs()
                .eval();
      for (Eigen::Index i = 0; i < y_err.size(); ++i) {
        EXPECT_LE(y_err[i], 2e-6);
      }
    }
    allocator.recover_memory();
  }
  if (!x_validated) {
    FAIL() << "Dense evaluation was never completed!";
  }
}

TEST_F(Riccati, nonosc_evolve_dense_output) {
  using namespace riccati::test;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::test::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<true, double>(omega_fun, gamma_fun, 16, 32,
                                                 32, 32);
  auto xi = 1e0;
  auto xf = 4e1;
  auto eps = 1e-12;
  auto epsh = 0.2;
  auto yi = airy_i(xi);
  auto dyi = airy_i_prime(xi);
  Eigen::Index Neval = 1e3;
  riccati::vector_t<double> x_eval
      = riccati::vector_t<double>::LinSpaced(Neval, xi, xf);
  auto ytrue
      = (airy_ai(-x_eval) + std::complex<double>(0.0, 1.0) * airy_bi(-x_eval))
            .eval();
  auto hi = 1.0 / omega_fun(xi);
  hi = choose_nonosc_stepsize(info, xi, hi, epsh);
  bool x_validated = false;
  while (xi < xf) {
    auto res
        = riccati::nonosc_evolve(info, xi, xf, yi, dyi, eps, epsh, hi, x_eval, allocator);
    if (!std::get<0>(res)) {
      break;
    } else {
      xi = std::get<1>(res);
      hi = std::get<2>(res);
      yi = std::get<1>(std::get<3>(res));
      dyi = std::get<2>(std::get<3>(res));
    }
    auto airy_true = airy_i(xi);
    auto airy_est = yi;
    auto err = std::abs((airy_true - airy_est) / airy_true);
    EXPECT_LE(err, 5e-9);
    auto start_y = std::get<5>(res);
    auto size_y = std::get<6>(res);
    if (size_y > 0) {
      x_validated = true;
      auto y_true_slice = ytrue.segment(start_y, size_y);
      auto&& y_est = std::get<4>(res);
      auto y_err
          = ((y_true_slice - y_est).array() / y_true_slice.array())
                .abs()
                .eval();
      for (int i = 0; i < y_err.size(); ++i) {
        EXPECT_LE(y_err[i], 6e-6);
      }
    }
    allocator.recover_memory();
  }
  if (!x_validated) {
    FAIL() << "Dense evaluation was never completed!";
  }
}

TEST_F(Riccati, evolve_dense_output_airy) {
  using namespace riccati::test;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::test::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<true, double>(omega_fun, gamma_fun, 16, 32,
                                                 32, 32);
  auto xi = 1e0;
  auto xf = 1e6;
  auto eps = 1e-12;
  auto epsh = 1e-13;
  auto yi = airy_i(xi);
  auto dyi = airy_i_prime(xi);
  Eigen::Index Neval = 1e3;
  riccati::vector_t<double> x_eval
      = riccati::vector_t<double>::LinSpaced(Neval, xi, 1e2);
  auto ytrue = airy_i(x_eval.array()).matrix().eval();
  auto res = riccati::evolve(info, xi, xf, yi, dyi, eps, epsh, 0.1, x_eval, allocator);
  auto y_err
      = ((std::get<6>(res) - ytrue).array() / ytrue.array()).abs().eval();
  for (int i = 0; i < y_err.size(); ++i) {
    EXPECT_LE(y_err[i], 9e-6) << "i = " << i << " x = " << x_eval[i] <<
      " y = " << ytrue[i] << " y_est = " << std::get<6>(res)[i];
  }
  EXPECT_LE(y_err.maxCoeff(), 9e-6);
}
TEST_F(Riccati, evolve_dense_output_burst) {
  using namespace riccati::test;
  constexpr int m = 1e6;
  auto omega_fun = [m](auto&& x) {
    return eval(
        matrix(riccati::test::sqrt(static_cast<double>(std::pow(m, 2)) - 1.0)
               / (1 + riccati::test::pow(array(x), 2.0))));
  };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<true, double>(omega_fun, gamma_fun, 16, 32,
                                                 32, 32);
  constexpr double xi = -m;
  constexpr double xf = m;
  auto burst_y = [m = static_cast<double>(m)](auto&& x) {
    return std::sqrt(1 + x * x) / m
           * (std::cos(m * std::atan(x))
              + std::complex<double>(0.0, 1.0) * std::sin(m * std::atan(x)));
  };
  auto yi = burst_y(xi);
  auto burst_dy = [mm = static_cast<double>(m)](auto&& x) {
    return (1.0 / std::sqrt(1.0 + x * x) / mm
            * ((x + std::complex<double>(0.0, 1.0) * mm)
                   * std::cos(mm * std::atan(x))
               + (-mm + std::complex<double>(0.0, 1.0) * x)
                     * std::sin(mm * std::atan(x))));
  };
  auto dyi = burst_dy(xi);
  auto eps = 1e-12;
  auto epsh = 1e-13;
  Eigen::Index Neval = 1e3;
  riccati::vector_t<double> x_eval
      = riccati::vector_t<double>::LinSpaced(Neval, xi, xf);
  auto res = riccati::evolve(info, xi, xf, yi, dyi, eps, epsh, 0.1, x_eval, allocator);
  auto x_steps = Eigen::Map<Eigen::VectorXd>(std::get<0>(res).data(), std::get<0>(res).size());
  auto ytrue = x_steps.unaryExpr(burst_y).eval();
  auto y_steps = Eigen::Map<Eigen::Matrix<std::complex<double>, -1, 1>>(std::get<1>(res).data(), std::get<1>(res).size());
  auto y_err
      = (((ytrue - y_steps).array()).abs() / (ytrue.array()).abs()).eval();
  // FRUSZINA: Doing dense evals here gives a max error of 0.0001 or so :(
  EXPECT_LE(y_err.maxCoeff(), 5e-9);
}

/**
[==========] Running 21 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 21 tests from Riccati
[ RUN      ] Riccati.chebyshev_coeffs_to_cheby_nodes_truth
[       OK ] Riccati.chebyshev_coeffs_to_cheby_nodes_truth (0 ms)
[ RUN      ] Riccati.chebyshev_cheby_nodes_to_coeffs_truth
[       OK ] Riccati.chebyshev_cheby_nodes_to_coeffs_truth (0 ms)
[ RUN      ] Riccati.coeffs_and_cheby_nodes
[       OK ] Riccati.coeffs_and_cheby_nodes (0 ms)
[ RUN      ] Riccati.chebyshev_integration_truth
[       OK ] Riccati.chebyshev_integration_truth (0 ms)
[ RUN      ] Riccati.quad_weights_test
[       OK ] Riccati.quad_weights_test (0 ms)
[ RUN      ] Riccati.chebyshev_chebyshev_truth
[       OK ] Riccati.chebyshev_chebyshev_truth (0 ms)
[ RUN      ] Riccati.chebyshev_integration
[       OK ] Riccati.chebyshev_integration (0 ms)
[ RUN      ] Riccati.interpolate_test
[       OK ] Riccati.interpolate_test (1 ms)
[ RUN      ] Riccati.spectral_chebyshev_test
[       OK ] Riccati.spectral_chebyshev_test (2 ms)
[ RUN      ] Riccati.osc_evolve_dense_output
[       OK ] Riccati.osc_evolve_dense_output (20 ms)
[ RUN      ] Riccati.nonosc_evolve_dense_output
[       OK ] Riccati.nonosc_evolve_dense_output (2641 ms)
[ RUN      ] Riccati.evolve_dense_output_burst
[       OK ] Riccati.evolve_dense_output_burst (265 ms)
[ RUN      ] Riccati.evolve_dense_output_airy
[       OK ] Riccati.evolve_dense_output_airy (15 ms)
[ RUN      ] Riccati.solver_make_solver_nondense
[       OK ] Riccati.solver_make_solver_nondense (1 ms)
[ RUN      ] Riccati.solver_make_solver_dense
[       OK ] Riccati.solver_make_solver_dense (1 ms)
[ RUN      ] Riccati.osc_step_test
[       OK ] Riccati.osc_step_test (1 ms)
[ RUN      ] Riccati.nonosc_step_test
[       OK ] Riccati.nonosc_step_test (1 ms)
[ RUN      ] Riccati.osc_stepsize_dense_output
[       OK ] Riccati.osc_stepsize_dense_output (1 ms)
[ RUN      ] Riccati.osc_stepsize_nondense_output
[       OK ] Riccati.osc_stepsize_nondense_output (1 ms)
[ RUN      ] Riccati.nonosc_stepsize_dense_output
[       OK ] Riccati.nonosc_stepsize_dense_output (1 ms)
[ RUN      ] Riccati.nonosc_stepsize_nondense_output
[       OK ] Riccati.nonosc_stepsize_nondense_output (1 ms)
[----------] 21 tests from Riccati (2958 ms total)

[----------] Global test environment tear-down
[==========] 21 tests from 1 test suite ran. (2958 ms total)
[  PASSED  ] 21 tests.
[sbronder@ccmlin064 build
*/

#include <riccati/solver.hpp>
#include <riccati/step.hpp>
#include <tests/cpp/utils.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string>

/**
 def test_osc_step():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = solversetup(w, g)
    x0 = 10.0
    h = 20.0
    epsres = 1e-12
    xscaled = x0 + h / 2 + h / 2 * info.xn
    info.wn = w(xscaled)
    info.gn = g(xscaled)
    y0 = sp.airy(-x0)[0]
    dy0 = -sp.airy(-x0)[1]
    y_ana = sp.airy(-(x0 + h))[0]
    dy_ana = -sp.airy(-(x0 + h))[1]
    y, dy, res, success, phase = osc_step(info, x0, h, y0, dy0, epsres=epsres)
    y_err = np.abs((y - y_ana) / y_ana)
    assert y_err < 1e-8 and res < epsres
*/
// TODO: Write these!!
TEST(riccati, osc_step_test) {}

/*
def test_nonosc_step():
    w = lambda x: np.sqrt(x)
    g = lambda x: np.zeros_like(x)
    info = solversetup(w, g)
    x0 = 1.0
    h = 0.5
    eps = 1e-12
    y0 = sp.airy(-x0)[2]
    dy0 = -sp.airy(-x0)[3]
    y, dy, err, success = nonosc_step(info, x0, h, y0, dy0, epsres=eps)
    y_ana = sp.airy(-x0 - h)[2]
    dy_ana = -sp.airy(-x0 - h)[3]
    y_err = np.abs((y - y_ana) / y_ana)
    dy_err = np.abs((dy - dy_ana) / dy_ana)
    assert y_err < 1e-8 and dy_err < 1e-8

*/
/**
 * template <typename SolverInfo, typename Scalar>
inline auto nonosc_step(SolverInfo &&info, Scalar x0, Scalar h,
                        std::complex<Scalar> y0, std::complex<Scalar> dy0,
                        Scalar epsres = Scalar(1e-12)) {
*/
TEST(riccati, nonosc_step_test) {
  using namespace riccati::test;
  auto omega_fun
      = [](auto&& x) { return eval(matrix(riccati::test::sqrt(array(x)))); };
  auto gamma_fun = [](auto&& x) { return zero_like(x); };
  auto info = riccati::make_solver<true, double>(omega_fun, gamma_fun, 16, 32,
                                                 32, 32);
  auto xi = 1e0;
  auto h = 0.5;
  auto xf = 1e6;
  auto eps = 1e-12;
  auto epsh = 1e-13;
  auto yi = airy_bi(-xi);
  auto dyi = -airy_bi_prime(-xi);
  riccati::print("yi", yi);
  riccati::print("dyi", dyi);
  auto res = riccati::nonosc_step(info, xi, h, yi, dyi, eps);
  auto y_ana = airy_ai(-xi - h);
  auto dy_ana = -airy_bi_prime(-xi - h);
  auto y_err = std::abs((std::get<1>(res) - y_ana) / y_ana);
  auto dy_err = std::abs((std::get<2>(res) - dy_ana) / dy_ana);
  std::cout << "y_err: " << y_err << std::endl;
  std::cout << "dy_err: " << dy_err << std::endl;

}

#include <iostream>
#include <fstream>

#include <nonlinfunc.hpp>
#include <autodiff.hpp>
#include <timestepper.hpp>

using namespace ASC_ode;

class PendulumAD : public NonlinearFunction
{
private:
  double m_length;
  double m_gravity;

public:
  PendulumAD(double length, double gravity = 9.81) : m_length(length), m_gravity(gravity) {}

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }

  void evaluate(VectorView<double> x, VectorView<double> f) const override
  {
    T_evaluate<double>(x, f);
  }

  void evaluateDeriv(VectorView<double> x, MatrixView<double> df) const override
  {
    Vector<AutoDiff<2>> x_ad(2);
    Vector<AutoDiff<2>> f_ad(2);

    x_ad(0) = Variable<0>(x(0));
    x_ad(1) = Variable<1>(x(1));
    T_evaluate<AutoDiff<2>>(x_ad, f_ad);

    for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 2; j++)
        df(i, j) = f_ad(i).deriv()[j];
  }

  template <typename T>
  void T_evaluate(VectorView<T> x, VectorView<T> f) const

  {
    f(0) = x(1);
    f(1) = -(m_gravity / m_length) * sin(x(0));
  }
};

class PendulumAD_2 : public NonlinearFunctionAutoDiff<PendulumAD_2>
{

private:
  double m_length;
  double m_gravity;

public:
  PendulumAD_2(double length, double gravity = 9.81) : m_length(length), m_gravity(gravity) {}

  template <typename T>
  void T_evaluate(VectorView<T> x, VectorView<T> f) const
  {
    f(0) = x(1);
    f(1) = -(m_gravity / m_length) * sin(x(0));
  }

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }
};

int main()
{
  double tend = 4 * M_PI;
  int steps = 100;
  double tau = tend / steps;

  Vector<> y = {1, 0}; // initializer list
  // auto rhs = std::make_shared<PendulumAD>(1.0, 1.0);
  auto rhs = std::make_shared<PendulumAD_2>(1.0, 1.0);
  // ExplicitEuler stepper(rhs);
  // ImprovedEuler stepper(rhs);
  //  ImplicitEuler stepper(rhs);
  CrankNicolson stepper(rhs);

  std::ofstream outfile("output_test_ode_ad.txt");
  std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  outfile << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  for (int i = 0; i < steps; i++)
  {
    stepper.doStep(tau, y);

    std::cout << (i + 1) * tau << "  " << y(0) << " " << y(1) << std::endl;
    outfile << (i + 1) * tau << "  " << y(0) << " " << y(1) << std::endl;
  }
}

#include <iostream>
#include <fstream>

#include <nonlinfunc.hpp>
#include <timestepper.hpp>

using namespace ASC_ode;
#define PI 3.141592653589793238463

class ElectricCircuit : public NonlinearFunction
{
  /*
  y (t) + R C y'(t) = U0(t)

  if now we use the time as variable

  y^prime = (U0(t) - y ) / (R C)
  t^pirme = 1

  out df is :

   ( -1/(R C)      U0'(t) )
       0           0  )

  */

private:
  double R;
  double C;

public:
  ElectricCircuit(double R, double C) : R(R), C(C) {}

  size_t dimX() const override
  {
    return 2;
  }
  size_t dimF() const override
  {
    return 2;
  }

  void evaluate(VectorView<double> y, VectorView<double> f) const override
  {

    f(0) = (std::cos(100 * PI * y(1)) - y(0)) / (R * C);
    f(1) = 1;
  }

  void evaluateDeriv(VectorView<double> y, MatrixView<double> df) const override
  {

    /*

       ( -1/(R C)      U0'(t) )
       0           0  )

    */
    df = 0.0;
    df(0, 0) = -1.0 / (R * C);
    df(0, 1) = -100 * PI * std::sin(100 * PI * y(1)) / (R * C);
  }
};

int main()
{
  double tend = 1.0 / 5; // 4 * M_PI;
  int steps = 10000;
  double tau = tend / steps;

  Vector<> y = {0, 0}; // initializer list
  auto rhs = std::make_shared<ElectricCircuit>(1, 1);
  ImplicitEuler stepper(rhs);
  // ImprovedEuler stepper(rhs);
  //  ImplicitEuler stepper(rhs);
  // CrankNicolson stepper(rhs);

  std::ofstream outfile("output_test_ode_electric.txt");
  std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  outfile << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  for (int i = 0; i < steps; i++)
  {
    stepper.doStep(tau, y);

    std::cout << (i + 1) * tau << "  " << y(0) << " " << y(1) << std::endl;
    outfile << (i + 1) * tau << "  " << y(0) << " " << y(1) << std::endl;
  }
}

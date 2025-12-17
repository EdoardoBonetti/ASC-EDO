#ifndef IMPLICITRK_HPP
#define IMPLICITRK_HPP

#include <vector.hpp>
#include <matrix.hpp>
#include <inverse.hpp>

namespace ASC_ode
{
  using namespace nanoblas;

  class ExplicitRungeKutta : public TimeStepper
  {
    Matrix<> m_a;
    Vector<> m_b, m_c;
    std::shared_ptr<NonlinearFunction> m_equ;
    std::shared_ptr<Parameter> m_tau;
    std::shared_ptr<ConstantFunction> m_yold;
    int m_stages;
    int m_n;
    Vector<> m_k, m_y;

  public:
    ExplicitRungeKutta(std::shared_ptr<NonlinearFunction> rhs,
                       const Matrix<> &a, const Vector<> &b, const Vector<> &c)
        : TimeStepper(rhs), m_a(a), m_b(b), m_c(c),
          m_tau(std::make_shared<Parameter>(0.0)),
          m_stages(c.size()), m_n(rhs->dimX()), m_k(m_stages * m_n), m_y(m_stages * m_n)
    {
      // Fist of all check that in the matrix j >= i and a(i,j) = 0
      for (int i = 0; i < m_stages; i++)
        for (int j = 0; j < i; j++)
          if (a(i, j) != 0.0)
            throw std::runtime_error("a(i,j) != 0 for i > j");

      // auto multiple_rhs = make_shared<MultipleFunc>(rhs, m_stages);
      // m_yold = std::make_shared<ConstantFunction>(m_stages * m_n);
      // auto knew = std::make_shared<IdentityFunction>(m_stages * m_n);
      //  m_equ = knew - Compose(multiple_rhs, m_yold + m_tau * std::make_shared<MatVecFunc>(a, m_n));
    }

    void doStep(double tau, VectorView<double> y) override
    {
      for (int j = 0; j < m_stages; j++)
        m_y.range(j * m_n, (j + 1) * m_n) = y;
      m_yold->set(m_y);

      m_tau->set(tau);
      m_k = 0.0;
      m_equ->evaluate(m_k, m_yold->get());
      for (int j = 0; j < m_stages; j++)
        y += tau * m_b(j) * m_k.range(j * m_n, (j + 1) * m_n);
    }
  };

  Matrix<double> Gauss2a{{0.25, 0.25 - sqrt(3) / 6}, {0.25 + sqrt(3) / 6, 0.25}};
  Vector<> Gauss2b{0.5, 0.5};
  Vector<> Gauss2c{0.5 - sqrt(3) / 6, 0.5 + sqrt(3) / 6};

  Vector<> Gauss3c{0.5 - sqrt(15) / 10, 0.5, 0.5 + sqrt(15) / 10};

  // codes from Numerical Recipes, https://numerical.recipes/book.html

  // Gauss integration on [0,1]

  /*
    given Runge-Kutta nodes c, compute the coefficients a and b
  */

}

#endif // IMPLICITRK_HPP
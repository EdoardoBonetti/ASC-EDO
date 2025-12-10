#ifndef AUTODIFF_HPP
#define AUTODIFF_HPP

#include <cstddef>
#include <ostream>
#include <cmath>
#include <array>
#include "nonlinfunc.hpp"

namespace ASC_ode
{

  template <size_t N, typename T = double>
  class Variable
  {
  private:
    T m_val;

  public:
    Variable(T v) : m_val(v) {}
    T value() const { return m_val; }

    // minus
  };

  template <typename T = double>
  auto derivative(T v, size_t /*index*/) { return T(0); }

  template <size_t N, typename T = double>
  class AutoDiff
  {
  private:
    T m_val;
    std::array<T, N> m_deriv;

  public:
    AutoDiff() : m_val(0), m_deriv{} {}
    AutoDiff(T v) : m_val(v), m_deriv{}
    {
      for (size_t i = 0; i < N; i++)
        m_deriv[i] = derivative(v, i);
    }

    template <size_t I>
    AutoDiff(Variable<I, T> var) : m_val(var.value()), m_deriv{}
    {
      m_deriv[I] = 1.0;
    }

    T value() const { return m_val; }
    std::array<T, N> &deriv() { return m_deriv; }
    const std::array<T, N> &deriv() const { return m_deriv; }
  };

  template <size_t N, typename T = double>
  auto derivative(AutoDiff<N, T> v, size_t index)
  {
    return v.deriv()[index];
  }

  template <size_t N, typename T>
  std::ostream &operator<<(std::ostream &os, const AutoDiff<N, T> &ad)
  {
    os << "Value: " << ad.value() << ", Deriv: [";
    for (size_t i = 0; i < N; i++)
    {
      os << ad.deriv()[i];
      if (i < N - 1)
        os << ", ";
    }
    os << "]";
    return os;
  }

  template <size_t N, typename T = double>
  AutoDiff<N, T> operator+(const AutoDiff<N, T> &a, const AutoDiff<N, T> &b)
  {
    AutoDiff<N, T> result(a.value() + b.value());
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = a.deriv()[i] + b.deriv()[i];
    return result;
  }
  template <size_t N, typename T = double>
  AutoDiff<N, T> operator-(const AutoDiff<N, T> &a, const AutoDiff<N, T> &b)
  {
    AutoDiff<N, T> result(a.value() - b.value());
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = a.deriv()[i] - b.deriv()[i];
    return result;
  }

  template <size_t N, typename T = double>
  auto operator+(T a, const AutoDiff<N, T> &b) { return AutoDiff<N, T>(a) + b; }

  template <size_t N, typename T = double>
  auto operator-(T a, const AutoDiff<N, T> &b) { return AutoDiff<N, T>(a) - b; }

  template <size_t N, typename T = double>
  AutoDiff<N, T> operator*(const AutoDiff<N, T> &a, const AutoDiff<N, T> &b)
  {
    AutoDiff<N, T> result(a.value() * b.value());
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = a.deriv()[i] * b.value() + a.value() * b.deriv()[i];
    return result;
  }

  template <size_t N, typename T = double>
  auto operator*(T a, const AutoDiff<N, T> &b) { return AutoDiff<N, T>(a) * b; }

  template <size_t N, typename T = double>
  AutoDiff<N, T> operator/(const AutoDiff<N, T> &a, const AutoDiff<N, T> &b)
  {

    AutoDiff<N, T> result(a.value() / b.value());
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = (a.deriv()[i] * b.value() - a.value() * b.deriv()[i]) / (b.value() * b.value());
    return result;
  }

  using std::cos;
  using std::exp;
  using std::sin;

  template <size_t N, typename T = double>
  AutoDiff<N, T> sin(const AutoDiff<N, T> &a)
  {
    AutoDiff<N, T> result(sin(a.value()));
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = cos(a.value()) * a.deriv()[i];
    return result;
  }
  template <size_t N, typename T = double>
  AutoDiff<N, T> cos(const AutoDiff<N, T> &a)
  {
    AutoDiff<N, T> result(sin(a.value()));
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = -sin(a.value()) * a.deriv()[i];
    return result;
  }
  template <size_t N, typename T = double>
  AutoDiff<N, T> exp(const AutoDiff<N, T> &a)
  {
    AutoDiff<N, T> result(exp(a.value()));
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = exp(a.value()) * a.deriv()[i];
    return result;
  }

  template <typename T>
  void LegendrePolynomialsInline(int n, T x, std::vector<T> &P)
  {
    if (n < 0)
    {
      P.clear();
      return;
    }
    P.resize(n + 1);
    P[0] = T(1);
    if (n == 0)
      return;
    P[1] = x;
    for (int k = 2; k <= n; ++k)
    {
      P[k] = ((T(2 * k - 1) * x * P[k - 1]) - T(k - 1) * P[k - 2]) / T(k);
    }
  }

  class NonlinearFunctionNumDiff : public NonlinearFunction
  {
    double m_eps;

  public:
    NonlinearFunctionNumDiff(double eps) : m_eps(eps) {}
    void evaluateDeriv(VectorView<double> x, MatrixView<double> df) const override
    {
      Vector xl(dimX()), xr(dimX());
      Vector fl(dimF()), fr(dimF());
      for (int i = 0; i < dimX(); i++)
      {
        xl = x;
        xr = x;
        xl(i) -= m_eps;
        xr(i) += m_eps;
        evaluate(xl, fl);
        evaluate(xr, fr);
        df.col(i) = 1 / (2 * m_eps) * (fr - fl);
      }
    }
  };

  template <typename NLF>
  class NonlinearFunctionAutoDiff : public NonlinearFunction
  {
  public:
    void evaluate(VectorView<double> x, VectorView<double> f) const override
    {
      static_cast<const NLF *>(this)->T_evaluate(x, f);
    }

    void evaluateDeriv(VectorView<double> x, MatrixView<double> df) const override
    {
      Vector<AutoDiff<1>> adx(dimX());
      Vector<AutoDiff<1>> adf(dimF());

      for (int i = 0; i < dimX(); i++)
      {
        for (int j = 0; j < dimX(); j++)
          adx(j) = x(j);
        adx(i) = Variable<0>(x(i));
        static_cast<const NLF *>(this)->T_evaluate(adx, adf);
        for (int j = 0; j < dimF(); j++)
          df(j, i) = adf(j).deriv()[0];
      }
    }
  };

} // namespace ASC_ode

#endif

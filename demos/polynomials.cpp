#include <iostream>
#include <autodiff.hpp>

using namespace ASC_ode;

int main(int argc, char **argv)
{
    int N = 1000;
    double x = -1;
    int poly_order = 5;
    // Create a double x in the interval -1 1 with step size 1/N
    for (int i = 0; i < N + 1; i++)
    {
        x = -1 + i * 2.0 / N;
        AutoDiff<1> adx = Variable<0>(x);
        std::vector<AutoDiff<1>> P;
        LegendrePolynomialsInline(poly_order, adx, P);

        std::cout << P[poly_order].value() << " ";

        std::cout << std::endl;
    }
}
#include "PEPPER.hpp"

int main()
{
    std::string eventFile = "gg2ttgg_10k.lhe";
    auto mesVector = PEP::PER::matrixCalculation( eventFile );
    std::cout << "\n\n" << mesVector[9999] << "\n\n";

    return 0;
}
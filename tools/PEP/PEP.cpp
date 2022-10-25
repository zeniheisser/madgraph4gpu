#include "PEP.hpp"
#include <time.h>

int main(){

std::string eventFileName = "gg2ttgg_10k.lhe";



std::cout.precision(10);




clock_t start1 = clock();
std::vector<double> eventVector = PEP::eventExtraction(eventFileName);
clock_t end1 = clock();

std::cout << std::fixed <<  float(end1 - start1)/CLOCKS_PER_SEC  << "\n";

return 0;

}
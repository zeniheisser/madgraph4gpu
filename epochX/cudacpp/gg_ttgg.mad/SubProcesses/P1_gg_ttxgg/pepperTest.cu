#include "PEPPER.hpp"

int main()
{
    std::string eventFile = "gg2ttgg_10k.lhe";
    auto mesVector = PEP::PER::matrixCalculation( eventFile );
    std::cout << "\n\n" << mesVector[9999] << "\n";

    auto parametercard = PEP::PER::filePuller("../../Cards/param_card.dat");
    std::cout << "\npulled param card\n";
    PEP::PER::paramReplacer( "../../Cards/param_card.dat", parametercard );
    std::cout << "\nreplaced param card\n";
    auto nuParams = PEP::PER::filePuller( "../../Cards/param_card.dat" );
    if(parametercard == nuParams){
        std::cout << "\nboth parameter cards are identical!\n";
    } else{
        std::cout <<"\nTHE PARAMCARDS DIFFER\n" << parametercard << "\n\n" << nuParams << "\n\n";
    }

    auto rwgtparams = PEP::PER::singleRwgtReader( "reweight_card.dat" );

    std::cout << "\n" << rwgtparams << "\n\n";

    return 0;
}
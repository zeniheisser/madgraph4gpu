#include "PEPPER.hpp"

int main()
{
    std::string eventFile = "gg2ttgg_10k.lhe";
    auto mesVector = PEP::PER::matrixCalculation( eventFile );
    std::cout << "\n\n" << mesVector[9999] << "\n";

    std::string fileLoc = "../../Cards/param_card.dat";

    auto parametercard = PEP::PER::filePuller(fileLoc);
    std::cout << "\npulled param card\n";
    PEP::PER::paramReplacer( fileLoc, parametercard );
    std::cout << "\nreplaced param card\n";
    auto nuParams = PEP::PER::filePuller( fileLoc );
    if(parametercard == nuParams){
        std::cout << "\nboth parameter cards are identical!\n";
    } else{
        std::cout <<"\nTHE PARAMCARDS DIFFER\n" << parametercard << "\n\n" << nuParams << "\n\n";
    }

    const std::string rwgtCard = PEP::PER::filePuller( "reweight_card.dat" );
    std::cout << "\n\n" << rwgtCard << "\n\n";

    auto rwgtparams = PEP::PER::singleRwgtReader( rwgtCard );

    std::cout << "\n" << rwgtparams << "\n\n";

    return 0;
}
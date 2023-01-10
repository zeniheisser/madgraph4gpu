#include "PEPPER.hpp"

int main()
{
    std::string eventFile = "gg2ttgg_10k.lhe";
    auto mesVector = PEP::PER::matrixCalculation( eventFile );
    std::cout << "\n\n" << mesVector[9999] << "\n";

    std::string fileLoc = "../../Cards/param_card.dat";

    auto parametercard = PEP::PER::filePuller(fileLoc);
    PEP::PER::paramReplacer( fileLoc, parametercard );
    auto nuParams = PEP::PER::filePuller( fileLoc );
    if(parametercard == nuParams){
        std::cout << "\nboth parameter cards are identical!\n";
    } else{
        std::cout <<"\nTHE PARAMCARDS DIFFER\n" << parametercard << "\n\n" << nuParams << "\n\n";
    }

    std::string rwgtCard = PEP::PER::filePuller( "reCard.dat" );

    auto rwgtparams = PEP::PER::singleRwgtReader( rwgtCard );

    std::cout << "\n" << rwgtparams << "\n\n";

    auto rwgtvector = PEP::PER::rwgtReader( rwgtCard );
    /* for(std::string parSet : rwgtvector){
        std::cout << "\n" << parSet << "\n";
    } */

    auto parameterCard = PEP::PER::filePuller( "../../Cards/param_card.dat" );

    auto nuParamCard = PEP::PER::paramCardReplacer( rwgtvector[0], parameterCard );

    auto tuParamCard = PEP::PER::paramCardReplacer( rwgtvector[1], parameterCard );

    std::cout << "\n\n" << nuParamCard << "\n\n" << tuParamCard << "\n\n";


    for( auto rwgtsets : rwgtvector ){
        std::cout << "\n\n" << rwgtsets << "\n\n";
    }

    return 0;
}
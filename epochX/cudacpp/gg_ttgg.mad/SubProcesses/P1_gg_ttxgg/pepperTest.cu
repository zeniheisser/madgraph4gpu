#include "PEPPER.hpp"

int main()
{
    /* std::string eventFile = "gg2ttgg_10k.lhe";
    //auto mesVector = PEP::PER::matrixCalculation( eventFile );
    //std::cout << "\n\n" << mesVector[9999] << "\n";

    std::string cardLoc = "../../Cards/param_card.dat";

    std::string rwgtLoc = "reCard.dat";

    auto parametercard = PEP::PER::filePuller(cardLoc);
    //PEP::PER::paramReplacer( fileLoc, parametercard );
    /* auto nuParams = PEP::PER::filePuller( fileLoc );
    if(parametercard == nuParams){
        std::cout << "\nboth parameter cards are identical!\n";
    } else{
        std::cout <<"\nTHE PARAMCARDS DIFFER\n" << parametercard << "\n\n" << nuParams << "\n\n";
    } 

    std::string rwgtCard = PEP::PER::filePuller( "reCard.dat" );

    auto rwgtparams = PEP::PER::singleRwgtReader( rwgtCard );

    std::cout << "\n" << rwgtparams << "\n\n";

    auto rwgtvector = PEP::PER::rwgtReader( rwgtCard );
    for(std::string parSet : rwgtvector){
        std::cout << "\n" << parSet << "\n";
    } 

    //std::string cardLoc = "../../Cards/param_card.dat";

    auto parameterCard = PEP::PER::filePuller( cardLoc );

    auto nuParamCard = PEP::PER::paramCardReplacer( rwgtvector[0], parameterCard );

    auto tuParamCard = PEP::PER::paramCardReplacer( rwgtvector[1], parameterCard );

    PEP::PER::filePusher( cardLoc, nuParamCard );
    auto resVector = PEP::PER::matrixCalculation( eventFile );
    PEP::PER::filePusher( cardLoc, tuParamCard );
    auto tesVector = PEP::PER::matrixCalculation( eventFile );
    PEP::PER::filePusher( cardLoc, parameterCard );

    std::cout << "\nMEs are   " << resVector[1000] << " and " << tesVector[1000] << "\n\n";

    auto nuWgts = PEP::PER::rwgtRunner( eventFile, rwgtLoc, cardLoc );
    for( auto wgtPtrs : nuWgts )
    {
        std::cout << "\n\n" << wgtPtrs->at(0) << "\n";
        std::cout << "\n\n" << wgtPtrs->at(10) << "\n";
        std::cout << "\n\n" << wgtPtrs->at(110) << "\n";
        std::cout << "\n\n" << wgtPtrs->at(1230) << "\n";
        std::cout << "\n\n" << wgtPtrs->at(3210) << "\n";
        std::cout << "\n\n" << wgtPtrs->at(6420) << "\n";
        std::cout << "\n\n" << wgtPtrs->at(1946) << "\n";
        std::cout << "\n\n" << wgtPtrs->at(9990) << "\n";
        std::cout << "\n\n" << wgtPtrs->at(5050) << "\n";
    } */

    auto attempVec = PEP::lheParser( "testLHE.lhe" );
    int totWgts = 0;
    for( int k = 2 ; k < attempVec.size() ; k = k+3 )
    {
        totWgts += attempVec[k]->size();
    }

    std::cout << "\n\nattVec has elems  " << attempVec.size() << "   and tot no wgts is   " << totWgts << "\n\n";


    return 0;
}
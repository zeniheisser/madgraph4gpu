// ZW: header for the PEPPER library,
// which interfaces with the cudacpp output
// program from vectorised port of MadGraph
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <chrono>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mgOnGpuConfig.h"

#include "BridgeKernels.h"
#include "CPPProcess.h"
#include "CrossSectionKernels.h"
#include "MatrixElementKernels.h"
#include "MemoryAccessMatrixElements.h"
#include "MemoryAccessMomenta.h"
#include "MemoryAccessRandomNumbers.h"
#include "MemoryAccessWeights.h"
#include "MemoryBuffers.h"
#include "RamboSamplingKernels.h"
#include "RandomNumberKernels.h"
#include "epoch_process_id.h"
#include "timermap.h"
#include "PEP.hpp"
#include "fbridge.cc"

namespace PEP::PER
{
    std::vector<double>& matrixCalculation( const std::string& lheFile )
    {
        auto vecPtr = PEP::lheParser( lheFile );
        const int nEvt = vecPtr[1]->size();
        static std::vector<double> meVector( nEvt );
        const unsigned int chanId = 0;
        const int nMom = 4;
        const int nPrts = (vecPtr[0]->size()) / ( nMom * nEvt );
        CppObjectInFortran *fortrPoint;
        fbridgecreate_( &fortrPoint, &nEvt, &nPrts, &nMom );
        fbridgesequence_( &fortrPoint, &vecPtr[0]->at(0), &vecPtr[1]->at(0), &meVector[0], &chanId );
        fbridgedelete_( &fortrPoint );

        return meVector;
    }

    std::string filePuller( const std::string& fileLoc )
    {
        std::ifstream fileLoad( fileLoc );
        std::stringstream buffer;
        buffer << fileLoad.rdbuf();
        std::string fileContent = buffer.str();
        std::transform( fileContent.begin(), fileContent.end(), fileContent.begin(), ::tolower );
        buffer.str(std::string());
        return fileContent;
    }

    void paramReplacer( const std::string& parCardLoc, const std::string& paramCard )
    {
        const char *parChardLoc = parCardLoc.c_str();
        auto remCheck = remove( parChardLoc );
        std::ofstream outputCard( parCardLoc );
        outputCard << paramCard;
        outputCard.close();
    }

    std::string& singleRwgtReader( std::string rwgtCard )
    {
        //std::transform( rwgtCard.begin(), rwgtCard.end(), rwgtCard.begin(), ::tolower );
        auto setPos = rwgtCard.find("set");
        auto firstLaunch = rwgtCard.find("\nlaunch", setPos);
        auto nuLine = rwgtCard.find("\n", setPos);
        static std::string rwgtParams = "";
        while( setPos < firstLaunch )
        {
            if( setPos == std::string::npos ){
                break;
            }
            rwgtParams += rwgtCard.substr(setPos + 4, nuLine - setPos - 4) + "\n";
            setPos = rwgtCard.find("set", nuLine);
            nuLine =  rwgtCard.find("\n", setPos);
        }
        return rwgtParams;
    }
    
    std::vector<std::string>& rwgtReader( std::string rwgtCard )
    {
        //std::transform( rwgtCard.begin(), rwgtCard.end(), rwgtCard.begin(), ::tolower );
        auto setPos = rwgtCard.find("set");
        auto launchPos = rwgtCard.find("\nlaunch", setPos);
        auto nuLine = rwgtCard.find("\n", setPos);
        static std::vector<std::string> rwgtParams;
        while( launchPos != std::string::npos )
        {
            auto firstLaunch = rwgtCard.find("\nlaunch", setPos);
            std::string rwgtParamStr = "";
            while ( setPos < firstLaunch)
            {
                if( setPos == std::string::npos ){
                    break;
                }
                rwgtParamStr += rwgtCard.substr(setPos + 4, nuLine - setPos - 4) + "\n";
                setPos = rwgtCard.find("set", nuLine);
                nuLine = rwgtCard.find("\n", setPos);
            }
            rwgtParams.push_back(rwgtParamStr);
            launchPos = rwgtCard.find("\nlaunch", setPos);
        }
        return rwgtParams;
    }

    std::vector<std::string>& splitByLine( const std::string& parameterSet )
    {
        static std::vector<std::string> lines;
        boost::split(lines, parameterSet, boost::is_any_of("\n"));
        return lines;
    }

    std::vector<std::string>& splitByBlank( const std::string& parameterLine )
    {
        static std::vector<std::string> words;
        boost::split( words, parameterLine, boost::is_any_of(" "));
        return words;
    }

    std::vector<int>& findBlockPar( std::vector<std::string> paramLine, std::string paramCard )
    {
        for( std::string lineWords : paramLine )
        {
            std::cout << "\n" << lineWords << "\n";
        }
        static std::vector<int> blockPars;
        auto blockLock = paramCard.find("block " + paramLine[0]);
        std::cout << "\nblockLock is  " << blockLock << "\n";
        if( paramLine[1] != "all" )
        {
            std::cout << "\nin if\n";
            auto paraLock = paramCard.find(" " + paramLine[2] + " ", blockLock );
            blockPars.push_back(paraLock + paramLine[2].length() + 2);
        } else 
        {
            std::cout << "\nin else\n";
            auto nuLine = paramCard.find( "\n", blockLock );
            auto blockEnd = paramCard.find( "###", blockLock );
            std::cout << "\nnuLine is   " << nuLine << "   and blockEnd is   " << blockEnd << "\n"; 
            while( nuLine < blockEnd )
            {
                auto tuLine = paramCard.find( "\n", nuLine + 1 );
                auto parNam = paramCard.find_first_not_of( " ", nuLine + 1 );
                auto parSpac = paramCard.find( " ", parNam );
                auto parPlac = paramCard.find_first_not_of( " ", parSpac );
                if( tuLine < blockEnd ){
                    blockPars.push_back( parPlac );
                }
                nuLine = tuLine;
            }
        }
        for( int vals : blockPars )
        {
            std::cout << "\n" << vals << "\n";
        }
        return blockPars;
    }
    
    std::vector<int>& findParamEnds( std::vector<int> blockParLocs, std::string paramCard )
    {
        static std::vector<int> lineEnds(blockParLocs.size());
        for( int k = 0; k < blockParLocs.size(); ++k)
        {
            lineEnds[k] = std::min(paramCard.find( "\n", blockParLocs[k] ), paramCard.find( " ", blockParLocs[k] ));
        }
        return lineEnds;
    }

    std::string& replaceBlockPar( std::vector<std::string> paramLine, std::string paramCard)
    {
        //REPLACE SINGLE SPECIFIC PARAMETER IN PARAMCARD
        auto parLocs = findBlockPar( paramLine, paramCard );
        static std::string modCard = paramCard.substr(0, parLocs[0] - 1);
        unsigned int srtPos = 0;
        auto endLocs = findParamEnds( parLocs, paramCard );
        for( int k = 0; k < parLocs.size(); ++k )
        {
            modCard +=  paramCard.substr( srtPos, parLocs[k] - srtPos ) + paramLine[2];
            srtPos = endLocs[k];
        }
        modCard += paramCard.substr(srtPos);
        return modCard;
    }

    /* RETURNTYPE paramCardReplacer()
    {
        REPLACE ALL GIVEN PARAMETERS IN PARAMCARD
    } */

}
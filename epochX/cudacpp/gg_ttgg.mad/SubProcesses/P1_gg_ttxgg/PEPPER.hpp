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

    std::string& singleRwgtReader( const std::string& rwgtCard )
    {
        std::string nuRwgtCard;
        std::transform( rwgtCard.begin(), rwgtCard.end(), nuRwgtCard.begin(), ::tolower );
        auto setPos = nuRwgtCard.find("set");
        auto firstLaunch = nuRwgtCard.find("\nlaunch", setPos);
        auto nuLine = nuRwgtCard.find("\n", setPos);
        static std::string rwgtParams = "";
        while( setPos < firstLaunch )
        {
            if( setPos == std::string::npos ){
                break;
            }
            rwgtParams += nuRwgtCard.substr(setPos + 4, nuLine - setPos - 4) + "\n";
            setPos = nuRwgtCard.find("set", nuLine);
            nuLine =  nuRwgtCard.find("\n", setPos);
        }
        return rwgtParams;
    }
    
    std::vector<std::string>& rwgtReader( std::string rwgtCard )
    {
        std::string nuRwgtCard;
        std::transform( rwgtCard.begin(), rwgtCard.end(), nuRwgtCard.begin(), ::tolower );
        auto setPos = nuRwgtCard.find("set");
        auto launchPos = nuRwgtCard.find("\nlaunch", setPos);
        auto nuLine = nuRwgtCard.find("\n", setPos);
        static std::vector<std::string> rwgtParams;
        while( launchPos != std::string::npos )
        {
            auto firstLaunch = nuRwgtCard.find("\nlaunch", setPos);
            std::string rwgtParamStr = "";
            while ( setPos < firstLaunch)
            {
                if( setPos == std::string::npos ){
                    break;
                }
                rwgtParamStr += nuRwgtCard.substr(setPos + 4, nuLine - setPos - 4) + "\n";
                setPos = nuRwgtCard.find("set", nuLine);
                nuLine = nuRwgtCard.find("\n", setPos);
            }
            rwgtParams.push_back(rwgtParamStr);
            launchPos = nuRwgtCard.find("\nlaunch", setPos);
        }
        return rwgtParams;
    }

}
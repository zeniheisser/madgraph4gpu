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
    std::vector<double>& matrixCalculation( std::string lheFile )
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

    std::string filePuller( std::string fileLoc )
    {
        std::ifstream fileLoad( fileLoc );
        std::stringstream buffer;
        buffer << fileLoad.rdbuf();
        std::string fileContent = buffer.str();
        buffer.str(std::string());
        return fileContent;
    }

    void paramReplacer( std::string& parCardLoc, std::string paramCard )
    {
        const char *parChardLoc = parCardLoc.c_str();
        auto remCheck = remove( parChardLoc );
        std::ofstream outputCard( parCardLoc );
        outputCard << paramCard;
        outputCard.close();
    }

    std::string& singleRwgtReader( const std::string& rwgtCard )
    {
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
    
    std::vector<std::string>>& rwgtReader( const std::string& rwgtCard )
    {
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

}
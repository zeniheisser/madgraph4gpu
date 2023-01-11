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
    std::vector<double> matrixCalculation( const std::string& lheFile )
    {
        auto vecPtr = PEP::lheParser( lheFile );
        const int nEvt = vecPtr[1]->size();
        std::vector<double> meVector( nEvt );
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

    void filePusher( const std::string& fileLoc, const std::string& fileCont )
    {
        std::ofstream fileWrite( fileLoc );
        fileWrite << fileCont;
        fileWrite.close();
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
    
    std::vector<std::string> rwgtReader( const std::string& rwgtCard )
    {
        //std::transform( rwgtCard.begin(), rwgtCard.end(), rwgtCard.begin(), ::tolower );
        auto setPos = rwgtCard.find("set");
        auto launchPos = rwgtCard.find("\nlaunch", setPos);
        auto nuLine = rwgtCard.find("\n", setPos);
        std::vector<std::string> rwgtParams;
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

    std::vector<std::string> splitByLine( const std::string& parameterSet )
    {
        std::vector<std::string> lines;
        boost::split(lines, parameterSet, boost::is_any_of("\n"));
        //vec.erase(std::remove(vec.begin(), vec.end(), 8), vec.end());
        lines.erase(std::remove(lines.begin(), lines.end(), ""), lines.end());
        return lines;
    }

    std::vector<std::string> splitByBlank( const std::string& parameterLine )
    {
        std::vector<std::string> words;
        boost::split( words, parameterLine, boost::is_any_of(" "));
        words.erase(std::remove(words.begin(), words.end(), ""), words.end());
        return words;
    }

    std::vector<int> findBlockPar( const std::vector<std::string>& paramLine, const std::string& paramCard )
    {
        std::vector<int> blockPars;
        auto blockLock = paramCard.find("block " + paramLine[0]);
        if( paramLine[1] != "all" )
        {
            auto paraLock = paramCard.find(" " + paramLine[1] + " ", blockLock );
            blockPars.push_back(paraLock + paramLine[1].length() + 2);
        } else if( paramLine[1] == "all" )
        {
            auto nuLine = paramCard.find( "\n", blockLock + 5 );
            auto blockEnd = paramCard.find( "####", blockLock  + 5);
            while( nuLine < blockEnd )
            {
                auto tuLine = paramCard.find( "\n", nuLine + 1 );
                auto parNam = paramCard.find_first_not_of( " ", nuLine + 2 );
                auto parSpac = paramCard.find( " ", parNam + 1 );
                auto parPlac = paramCard.find_first_not_of( " ", parSpac +1 );
                if( tuLine < blockEnd ){
                    blockPars.push_back( parPlac );
                }
                nuLine = tuLine;
            }
        }
        return blockPars;
    }
    
    std::vector<int> findParamEnds( const std::vector<int>& blockParLocs, const std::string& paramCard )
    {
        std::vector<int> lineEnds(blockParLocs.size());
        for( int k = 0; k < blockParLocs.size(); ++k)
        {
            auto endLinePos = paramCard.find( "\n", blockParLocs[k] );
            auto endWordPos = paramCard.find( " ", blockParLocs[k] );
            if( endLinePos == endWordPos ){
                std::cout << "\n\nnext endl and blankspace have the same position, which should not be possible. check that\n\n";
                lineEnds[k] = std::string::npos;
                continue;
            } else if ( endWordPos == std::string::npos ) {
                lineEnds[k] = endLinePos;
                continue;
            } else if ( endLinePos == std::string::npos ) {
                lineEnds[k] = endWordPos;
                continue;
            }
            // ZW: making sure we don't accidentally append std::string::npos, which is defined as unsigned integer -1,
            // meaning for comparison depending on algo may be treated as smaller than proper unsigned integers
            lineEnds[k] = std::min(paramCard.find( "\n", blockParLocs[k] ), paramCard.find( " ", blockParLocs[k] ));
        }
        return lineEnds;
    }

    std::vector<int> findParamLines( const std::vector<int>& blockParLocs, const std::string& paramCard )
    {
        std::vector<int> lineEnds(blockParLocs.size());
        for( int k = 0; k < blockParLocs.size(); ++k)
        {
            lineEnds[k] = paramCard.find( "\n", blockParLocs[k] ) + 1;
        }
        return lineEnds;
    }

    std::vector<std::string> paramNameVec( const std::vector<int>& paramLocs, const std::string& paramCard )
    {
        std::vector<std::string> paramSet;
        for( auto parPos : paramLocs )
        {
            auto startPos = paramCard.rfind( "\n", parPos ) + 1;
            paramSet.push_back(paramCard.substr( startPos, parPos - startPos ));
        }
        return paramSet;
    }

    std::string replaceBlockPar( const std::vector<std::string>& paramLine, const std::string& paramCard)
    {
        auto parLocs = findBlockPar( paramLine, paramCard );
        auto parNames = paramNameVec( parLocs, paramCard );
        auto startPos = paramCard.rfind( "\n", parLocs[0] ) + 1;
        std::string modCard = paramCard.substr(0, startPos);
        unsigned int srtPos = 0;
        auto endLocs = findParamEnds( parLocs, paramCard );
        auto lineLocs = findParamLines( parLocs, paramCard );
        for( int k = 0; k < parLocs.size(); ++k )
        {
            modCard += parNames[k] + paramLine[2] + paramCard.substr( endLocs[k], lineLocs[k] - endLocs[k]);
        }
        modCard += paramCard.substr(lineLocs[lineLocs.size() - 1]);
        return modCard;
    }

    std::string paramCardReplacer( const std::string& paramSet, const std::string& paramCard )
    {
        std::string modiCard = paramCard;
        auto paramSetVec = splitByLine(paramSet);
        for( auto params : paramSetVec )
        {
            auto paramVec = splitByBlank( params );
            modiCard = replaceBlockPar( paramVec, modiCard );
        }
        return modiCard;
    }

    std::vector<double> rwgtRunner( const std::string& lheFile, const std::string& rwgtCard, const std::string& paramCard )
    {
        /* auto vecPtr = PEP::lheParser( lheFile );
        const int nEvt = vecPtr[1]->size();
        std::vector<double> meVector( nEvt );
        const unsigned int chanId = 0;
        const int nMom = 4;
        const int nPrts = (vecPtr[0]->size()) / ( nMom * nEvt );
        CppObjectInFortran *fortrPoint;
        fbridgecreate_( &fortrPoint, &nEvt, &nPrts, &nMom );
        fbridgesequence_( &fortrPoint, &vecPtr[0]->at(0), &vecPtr[1]->at(0), &meVector[0], &chanId );
        fbridgedelete_( &fortrPoint ); */

        auto vecPtr = PEP::lheParser( lheFile );
        auto origParams = PEP::PER::filePuller( paramCard );
        auto rwgtParams = rwgtReader( filePuller( rwgtCard ) );
        const int nEvt = vecPtr[1]->size();
        std::vector<std::vector<double>*> mesPtrVec;
        const unsigned int chanId = 0;
        const int nMom = 4;
        const int nPrts = (vecPtr[0]->size()) / ( nMom * nEvt );
        std::vector<double> ogMEs( nEvt );
        CppObjectInFortran *fortrPoint;
        fbridgecreate_( &fortrPoint, &nEvt, &nPrts, &nMom );
        fbridgesequence_( &fortrPoint, &vecPtr[0]->at(0), &vecPtr[1]->at(0), &ogMEs[0], &chanId );
        for( auto parSets : rwgtParams )
        {
        filePusher( paramCard, paramCardReplacer( parSets, origParams ) );
        for( int k = 0 ; k < vecPtr.size() ; k = k + 3 );
        {
            std::vector<double> meVector( nEvt );
            fbridgesequence_( &fortrPoint, &vecPtr[k]->at(0), &vecPtr[k+1]->at(0), &meVector[0], &chanId );
            mesPtrVec.push_back( &meVector );
        }
        }
        fbridgedelete_( &fortrPoint );
        filePusher( paramCard, origParams );
        std::vector<double>& ogWgts = *(vecPtr[2]);
        std::vector<std::vector<double>*> rwgtVecs( mesPtrVec.size() );
        for( int k = 0 ; k < mesPtrVec.size() ; ++k )
        {
        std::vector<double> nuWgts( nEvt );
        for( int m = 0 ; m < nEvt ; ++m )
        {
            nuWgts[m] = (((*mesPtrVec[k])[m]) / (ogMEs[m])) * ogWgts[k];
        }
        rwgtVecs[k] = &nuWgts;
        }
        return rwgtVecs;
    }

}
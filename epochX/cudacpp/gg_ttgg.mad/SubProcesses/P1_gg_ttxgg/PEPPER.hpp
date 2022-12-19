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
        const unsigned int nEvt = *(vecPtr[1]).size();
        static std::vector<double> meVector( *(vecPtr[1]).size() );
        const unsigned int chanId = 0;
        const int nMom = 4;
        const unsigned int nPrts = (*(vecPtr[0]).size()) / ( nMom * nEvt );
        CppObjectInFortran *fortrPoint;
        fbridgecreate_( &fortrPoint, &nEvt, &nPrts, &nMom );
        fbridgesequence_( &fortrPoint, &vecPtr[0]->at(0), &vecPtr[1]->at(0), &meVector[0], &chanId );
        fbridgedelete_( &fortrPoint );

        return meVector;
    }
}
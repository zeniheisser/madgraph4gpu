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

#ifdef __CUDACC__
  using namespace mg5amcGpu;
#else
  using namespace mg5amcCpu;
#endif


int main()
{

  std::vector<double> eventVector = PEP::eventExtraction("gg2ttgg_1024.lhe");

  const int nevt = eventVector[eventVector.size() - 1];
  const int nPrt = eventVector[eventVector.size() - 2];
  const int nMom = 4;


  std::vector<double> momVector( 4 * nevt * nPrt );
  std::vector<double> gsVector( nevt );

  // ZW: change eventVector ordering so that the momentum order for
  // each particle is (E, px, py, pz) instead of the LHEF convention
  // which is (px, py, pz, E)
  for( unsigned int ievt = 0; ievt < nevt; ++ievt )
  {
    for( unsigned int iprt = 0; iprt < nPrt; ++iprt)
    {
      momVector[4*6*ievt + 4*iprt] = eventVector[4*6*ievt + 4*iprt + 3];
      for( unsigned int imom = 0; imom < 3; ++imom)
      {
        momVector[4*6*ievt + 4*iprt + 3 - imom] = eventVector[4*6*ievt + 4*iprt + 2 - imom];
      }
    }
    gsVector[ ievt ] = eventVector[ 4 * nPrt * nevt + ievt ];
  }

  CppObjectInFortran *fortrPoint;
  std::vector<double> mesVector( nevt );
  const unsigned int chanId = 0;
  fbridgecreate_( &fortrPoint, &nevt, &nPrt, &nMom );
  fbridgesequence_( &fortrPoint, &momVector[0], &gsVector[0], &mesVector[0], &chanId );
  fbridgedelete_( &fortrPoint );

  for( auto matrElem : mesVector )
  {
    std::cout << matrElem << "\n";
  }



  return 0;
}
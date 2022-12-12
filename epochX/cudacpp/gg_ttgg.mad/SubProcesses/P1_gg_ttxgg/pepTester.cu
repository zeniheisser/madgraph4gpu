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

  std::vector<double> eventVector = PEP::eventExtraction("gg2ttgg_10k.lhe");

  const int nevt = eventVector[eventVector.size() - 1];
  const int nPrt = eventVector[eventVector.size() - 2];
  const int nMom = 4;
  const int nWarpRemain = (32 - ( nevt % 32 )) % 32;
  const int nEvtExt = nevt + nWarpRemain; 


  std::vector<double> momVector( 4 * nEvtExt * nPrt );
  std::vector<double> gsVector( nEvtExt );
  std::vector<double> wgtsVector( nEvtExt );

  // ZW: change eventVector ordering so that the momentum order for
  // each particle is (E, px, py, pz) instead of the LHEF convention
  // which is (px, py, pz, E)
  for( unsigned int ievt = 0; ievt < nevt; ++ievt )
  {
    for( unsigned int iprt = 0; iprt < nPrt; ++iprt)
    {
      momVector[4*nPrt*ievt + 4*iprt] = eventVector[4*nPrt*ievt + 4*iprt + 3];
      for( unsigned int imom = 0; imom < 3; ++imom)
      {
        momVector[4*nPrt*ievt + 4*iprt + 3 - imom] = eventVector[4*nPrt*ievt + 4*iprt + 2 - imom];
      }
    }
    gsVector[ ievt ] = eventVector[ 4 * nPrt * nevt + ievt ];
    wgtsVector[ ievt ] = eventVector[ (4 * nPrt + 1) * nevt + ievt];
  }

  for( unsigned int ievt = 0; ievt < nWarpRemain; ++ievt)
  {
    for( unsigned int iprt = 0; iprt < nPrt; ++iprt)
    {
      for( unsigned int imom = 0; imom < 4; ++imom)
      {
        momVector[4*nPrt*nevt +4*nPrt*ievt + 4*iprt + imom] = 0.;
      }
    }
    gsVector[nevt + ievt] = 0.;
    wgtsVector[nevt + ievt] = 0.;
  }

  CppObjectInFortran *fortrPoint;
  std::vector<double> mesVector( nEvtExt );
  const unsigned int chanId = 0;
  fbridgecreate_( &fortrPoint, &nEvtExt, &nPrt, &nMom );
  fbridgesequence_( &fortrPoint, &momVector[0], &gsVector[0], &mesVector[0], &chanId );
//  fbridgedelete_( &fortrPoint );

 /*  for( auto matrElem : mesVector )
  {
    std::cout << matrElem << "\n";
  }

  for( auto wgts : wgtsVector )
  {
    std::cout << wgts << "\n";
  }
 */


 std::vector<double> mesVector2( nEvtExt );
 for( unsigned int i = 0; i < nevt; ++i )
 {
    constexpr double fixedG = 1.2177157847767195; // fixed G for aS=0.118 (hardcoded for now in check_sa.cc, fcheck_sa.f, runTest.cc)
    gsVector[i] = fixedG;
    //if ( i > 0 ) hstGs[i] = 0; // try hardcoding G only for event 0
    //hstGs[i] = i;
 }

  fbridgesequence_( &fortrPoint, &momVector[0], &gsVector[0], &mesVector2[0], &chanId );
  fbridgedelete_( &fortrPoint );

 /*  for( unsigned int k = 0; k < nevt; ++k)
  {
    std::cout << "OG wgt is " << wgtsVector[k] << " and RWd is " << (mesVector2[k] / mesVector[k]) * wgtsVector[k] << "\n";
  }  */

  pt::ptree lheFile;

  try {
      pt::read_xml("gg2ttgg_10k.lhe", lheFile);
  } catch (pt::xml_parser_error &e) {
      std :: cout << "Failed to parse LHE file" << e.what();
  } catch (...) {        std :: cout << "Undefined error while parsing LHE file";
  }

  std::set<std::pair<std::string, int>> processSet = PEP::procExtractor(lheFile);
  //std::cout << "\n" << processSet.size() << "\n";

  for( std::pair<std::string, int> pairSet : processSet )
  {
    std::cout << "Process: " << pairSet.first << " with " << pairSet.second << " external particles\n";
  }

 /*  std::vector<std::string> eventElems = PEP::eventExtractor(lheFile);
  for( auto entry : eventElems ){
      std::cout << "Entry: " << entry << "\n";
  } */

  auto vecPtr = PEP::eventParser("gg2ttgg_10k.lhe");
  std::cout << vecPtr[2]->at(0) << "\n";


  return 0;
}
//==========================================================================
// This file has been automatically generated for CUDA/C++ standalone
// MadGraph5_aMC@NLO v. 2.9.5, 2021-08-22
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
// GPU Template
//==========================================================================

#ifndef HelAmps_sm_H
#define HelAmps_sm_H

#include <cmath>
#include "mgOnGpuConfig.h"
#include "mgOnGpuTypes.h"
#include "mgOnGpuVectors.h"

using namespace std;

namespace MG5_sm
{
  // [NB the helas.h template is currently empty - the following comes from get_header_txt]

  __device__ void VVV1P0_1( const cxtype V2[], const cxtype V3[], const cxtype COUP, const fptype M1, const fptype W1, cxtype V1[] );

  __device__ void FFV1_0( const cxtype F1[], const cxtype F2[], const cxtype V3[], const cxtype COUP, cxtype* vertex );

  __device__ void FFV1_1( const cxtype F2[], const cxtype V3[], const cxtype COUP, const fptype M1, const fptype W1, cxtype F1[] );

  __device__ void FFV1_2( const cxtype F1[], const cxtype V3[], const cxtype COUP, const fptype M2, const fptype W2, cxtype F2[] );

} // end namespace MG5_sm

#endif // HelAmps_sm_H

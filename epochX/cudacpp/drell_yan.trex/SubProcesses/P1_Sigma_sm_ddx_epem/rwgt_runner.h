//==========================================================================
// Copyright (C) 2023-2024 CERN
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Written by: Z. Wettersten (June 2024) for the MG5aMC CUDACPP plugin.
//==========================================================================
//==========================================================================
// This file has been automatically generated for the CUDACPP plugin by
// MadGraph5_aMC@NLO v. 3.6.0, 2024-09-30
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================
//==========================================================================
// A class for reweighting matrix elements for
// Process: d d~ > e+ e- WEIGHTED<=4 @1
  // Process: d d~ > mu+ mu- WEIGHTED<=4 @1
  // Process: s s~ > e+ e- WEIGHTED<=4 @1
  // Process: s s~ > mu+ mu- WEIGHTED<=4 @1
//--------------------------------------------------------------------------

#ifndef _P1_Sigma_sm_ddx_epem_RUNNER_H_
#define _P1_Sigma_sm_ddx_epem_RUNNER_H_

#include "rwgt_instance.h"

namespace P1_Sigma_sm_ddx_epem {

    std::shared_ptr<std::vector<FORTRANFPTYPE>> amp( int& nEvt, int& nPar, int& nMom, std::vector<FORTRANFPTYPE>& momenta, std::vector<FORTRANFPTYPE>& alphaS, std::vector<FORTRANFPTYPE>& rndHel, std::vector<FORTRANFPTYPE>& rndCol, std::vector<int>& selHel, std::vector<int>& selCol, int& chanId, bool& goodHel );
    rwgt::fBridge bridgeConstr( std::vector<REX::event>& process, unsigned int warpSize );
    rwgt::fBridge bridgeConstr();
    std::shared_ptr<std::vector<size_t>> procSort( std::string_view status, std::vector<std::string_view> arguments, size_t index = REX::npos );
    bool checkProc( REX::event& process, std::vector<std::string>& relStats );
    std::function<bool( REX::event& )> getComp();
    REX::eventSet eventSetConstruct( std::vector<REX::event>& process );
    REX::eventSet getEventSet();

}



#endif
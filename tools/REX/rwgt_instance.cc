//==========================================================================
// Copyright (C) 2023-2024 CERN
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Written by: Z. Wettersten (Jan 2024) for the MG5aMC CUDACPP plugin.
//==========================================================================
//==========================================================================
// Library including generic functions and classes for event reweighting.
// Process-specific rwgt_runner files are generated by mg5amc@nlo and use
// this library, while the rwgt_driver file is a wrapping program that 
// calls the process-specific runners for given subprocesses.
//==========================================================================

#ifndef _RWGT_INSTANCE_CC_
#define _RWGT_INSTANCE_CC_

#include "rwgt_instance.h"

namespace rwgt{

    //ZW: Function for calculating the number of remaining events in a warp
    // in order to pad the input arrays to a multiple of the warp size
    unsigned int warpRemain( unsigned int nEvt, unsigned int nWarp ){
        return (nWarp - ( nEvt % nWarp )) % nWarp;
    }

    //ZW: Function for padding the input arrays to a multiple of the warp size
    template<typename T>
    void warpPad( std::vector<T>& input, unsigned int nWarp = 32 ){
        auto nEvt = input.size();
        auto nWarpRemain = warpRemain( nEvt, nWarp );
        input.reserve( nEvt + nWarpRemain );
        for( size_t k = nEvt - nWarpRemain ; k < nEvt ; ++k ){
            input.push_back( input[k] );
        }
        return;
    }

    fBridge::fBridge(){}
    fBridge::fBridge( REX::event& process ){
        this->nPar = process.getPrts().size();
        this->goodHel = false;
    }
    fBridge::fBridge( std::vector<REX::event>& process, unsigned int warpSize){
        this->nPar = process[0].getPrts().size();
        this->nEvt = process.size();
        this->nWarp = warpSize;
        this->nWarpRemain = warpRemain( nEvt, nWarp );
        this->fauxNEvt = nEvt + nWarpRemain;
        this->rndHel = std::vector<FORTRANFPTYPE>( fauxNEvt, 0. );
        this->rndCol = std::vector<FORTRANFPTYPE>( fauxNEvt, 0. );
        this->selHel = std::vector<int>( fauxNEvt, 0. );
        this->selCol = std::vector<int>( fauxNEvt, 0. );
        this->goodHel = false;
    }
    fBridge::fBridge( std::vector<std::shared_ptr<REX::event>> process, unsigned int warpSize){
        this->nPar = process[0]->getPrts().size();
        this->nEvt = process.size();
        this->nWarp = warpSize;
        this->nWarpRemain = warpRemain( nEvt, nWarp );
        this->fauxNEvt = nEvt + nWarpRemain;
        this->rndHel = std::vector<FORTRANFPTYPE>( fauxNEvt, 0. );
        this->rndCol = std::vector<FORTRANFPTYPE>( fauxNEvt, 0. );
        this->selHel = std::vector<int>( fauxNEvt, 0. );
        this->selCol = std::vector<int>( fauxNEvt, 0. );
        this->goodHel = false;
    }
    fBridge::fBridge( const fBridge& source ){
        this->rndHel = source.rndHel;
        this->rndCol = source.rndCol;
        this->selHel = source.selHel;
        this->selCol = source.selCol;
        this->chanId = source.chanId;
        this->nMom = source.nMom;
        this->nWarp = source.nWarp;
        this->nWarpRemain = source.nWarpRemain;
        this->nEvt = source.nEvt;
        this->fauxNEvt = source.fauxNEvt;
        this->nPar = source.nPar;
        this->bridge = source.bridge;
        this->goodHel = source.goodHel;
    }
    void fBridge::init( std::vector<REX::event>& process, unsigned int warpSize ){
        this->nPar = process[0].getPrts().size();
        this->nEvt = process.size();
        this->nWarp = warpSize;
        this->nWarpRemain = warpRemain( nEvt, nWarp );
        this->fauxNEvt = nEvt + nWarpRemain;
        this->rndHel = std::vector<FORTRANFPTYPE>( fauxNEvt, 0. );
        this->rndCol = std::vector<FORTRANFPTYPE>( fauxNEvt, 0. );
        this->selHel = std::vector<int>( fauxNEvt, 0. );
        this->selCol = std::vector<int>( fauxNEvt, 0. );
    }
    void fBridge::init( std::vector<std::shared_ptr<REX::event>> process, unsigned int warpSize ){
        this->nPar = process[0]->getPrts().size();
        this->nEvt = process.size();
        this->nWarp = warpSize;
        this->nWarpRemain = warpRemain( nEvt, nWarp );
        this->fauxNEvt = nEvt + nWarpRemain;
        this->rndHel = std::vector<FORTRANFPTYPE>( fauxNEvt, 0. );
        this->rndCol = std::vector<FORTRANFPTYPE>( fauxNEvt, 0. );
        this->selHel = std::vector<int>( fauxNEvt, 0. );
        this->selCol = std::vector<int>( fauxNEvt, 0. );
    }
    void fBridge::bridgeSetup( unsigned int& noEvts, unsigned int warpSize ){
        this->nEvt = noEvts;
        this->nWarp = warpSize;
        this->nWarpRemain = warpRemain( nEvt, nWarp );
        this->fauxNEvt = nEvt + nWarpRemain;
        this->rndHel = std::vector<FORTRANFPTYPE>( fauxNEvt, 0. );
        this->rndCol = std::vector<FORTRANFPTYPE>( fauxNEvt, 0. );
        this->selHel = std::vector<int>( fauxNEvt, 0. );
        this->selCol = std::vector<int>( fauxNEvt, 0. );
    }
    void fBridge::bridgeSetup( std::vector<FORTRANFPTYPE>& evVec, unsigned int warpSize ){
        this->nEvt = evVec.size();
        this->nWarp = warpSize;
        this->nWarpRemain = warpRemain( nEvt, nWarp );
        this->fauxNEvt = nEvt + nWarpRemain;
        this->rndHel = std::vector<FORTRANFPTYPE>( fauxNEvt, 0. );
        this->rndCol = std::vector<FORTRANFPTYPE>( fauxNEvt, 0. );
        this->selHel = std::vector<int>( fauxNEvt, 0. );
        this->selCol = std::vector<int>( fauxNEvt, 0. );
    }
    void fBridge::bridgeSetup( std::shared_ptr<std::vector<FORTRANFPTYPE>>& evVec, unsigned int warpSize ){
        this->bridgeSetup( *evVec, warpSize );
    }
    void fBridge::setBridge( bridgeWrapper& amp ){
        if( this->bridge == nullptr){
            this->bridge = amp;
        } else throw std::runtime_error("fBridge object doubly defined.");
    }
    std::shared_ptr<std::vector<FORTRANFPTYPE>> fBridge::bridgeCall( std::vector<FORTRANFPTYPE>& momenta, std::vector<FORTRANFPTYPE>& alphaS ){
        if(this->nEvt == 0) this->bridgeSetup( alphaS );
        if( this->bridge == nullptr) throw std::runtime_error("fBridge object not defined.");
        warpPad( alphaS, nWarp );
        warpPad( momenta, nWarp * nPar * nMom );
        auto evalScatAmps = this->bridge(fauxNEvt, nPar, nMom, momenta, alphaS, rndHel, rndCol, selHel, selCol, chanId, goodHel );
        alphaS.resize( nEvt );
        momenta.resize( nEvt * nPar * nMom );
        evalScatAmps->resize( nEvt );
        return evalScatAmps;
    }

    instance::instance(){}
    instance::instance( std::vector<std::pair<int,int>>& event){
        this->procEventInt = event;
        this->process = REX::event( event );
    }
    instance::instance( std::vector<std::pair<int,int>>& event, REX::teaw::amplitude& amp ){
        this->procEventInt = event;
        this->process = REX::event( event );
        bridgeCall = amp;
    }
    void instance::setProc( std::vector<std::pair<int,int>>& event ){
        this->procEventInt = event;
        this->process = REX::event( event );
    }
    instance::instance( std::vector<std::pair<std::string,std::string>>& event){
        this->procEventStr = event;
        this->process = REX::event( event );
    }
    instance::instance( std::vector<std::pair<std::string,std::string>>& event, REX::teaw::amplitude& amp ){
        this->procEventStr = event;
        this->process = REX::event( event );
        bridgeCall = amp;
    }
    void instance::setProc( std::vector<std::pair<std::string,std::string>>& event ){
        this->procEventStr = event;
        this->process = REX::event( event );
    }
    void instance::setAmp( REX::teaw::amplitude& amp ){
        bridgeCall = amp;
    }
    std::shared_ptr<std::vector<FORTRANFPTYPE>> instance::ampEval( std::vector<double>& momenta, std::vector<double>& alphaS ){
        return bridgeCall( momenta, alphaS );
    }
    std::shared_ptr<std::vector<FORTRANFPTYPE>> instance::ampEval( std::shared_ptr<std::vector<double>> momenta, 
    std::shared_ptr<std::vector<double>> alphaS ){
        return bridgeCall( *momenta, *alphaS );
    }

}

#endif
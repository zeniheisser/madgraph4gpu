/***
 *     _                     ______ _______   __
 *    | |                    | ___ \  ___\ \ / /
 *    | |_ ___  __ ___      _| |_/ / |__  \ V / 
 *    | __/ _ \/ _` \ \ /\ / /    /|  __| /   \ 
 *    | ||  __/ (_| |\ V  V /| |\ \| |___/ /^\ \
 *     \__\___|\__,_| \_/\_/ \_| \_\____/\/   \/
 *                                              
 ***/
//
// *t*ensorial *e*vent *a*daption *w*ith *REX* Version 0.9.0
// teawREX is an extension to the REX C++ library for parsing and manipulating Les Houches Event-format (LHE) files,
// designed for leading order event reweighting based on input LHE file(s) and scattering amplitude functions.
// teawREX is in development and may not contain all features necessary for all desired features,
// and does not have documentation beyond the code itself.
//
// Copyright © 2023-2024 CERN, CERN Author Zenny Wettersten. 
// Licensed under the GNU Lesser General Public License (version 3 or later).
// All rights not expressly granted are reserved.
//

#ifndef _TEAWREX_CC_
#define _TEAWREX_CC_

#include <unistd.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <variant>
#include <stdarg.h>
#include "REX.h"
#include "teawREX.h"

namespace REX::teaw
{

    template<typename T1, typename T2>
    std::shared_ptr<std::vector<T1>> scatAmpEval(std::vector<T2>& momenta, std::function<std::shared_ptr<std::vector<T1>>(std::vector<T2>&)> evalFunc)
    { return evalFunc(momenta); }

    template<typename T1, typename T2>
    std::shared_ptr<std::vector<T1>> scatAmpEval(std::vector<T2>& momenta, std::function<std::vector<T1>(std::vector<T2>&)> evalFunc)
    { return evalFunc(momenta); }

    template<typename T1, typename T2>
    std::shared_ptr<std::vector<T1>> scatAmpEval(std::vector<T2>& momenta, std::function<std::shared_ptr<std::vector<T1>>(std::vector<T2>&, std::vector<T2>&)> evalFunc)
    { return evalFunc(momenta); }

    template<typename T1, typename T2>
    std::shared_ptr<std::vector<T1>> scatAmpEval(std::vector<T2>& momenta, std::function<std::vector<T1>(std::vector<T2>&, std::vector<T2>&)> evalFunc)
    { return evalFunc(momenta); }

        rwgtVal::rwgtVal() : paramVal(){ return; }
        rwgtVal::rwgtVal( std::string_view paramLine )
        : paramVal( paramLine, false ){if( paramLine.size() == 0 ){ return; }
            realLine = paramLine;
            auto vals = *REX::blankSplitter( realLine );
            blockName = vals[1];
            idStr = vals[2];
            valStr = vals[3];
        }
        std::string_view rwgtVal::getLine(){ return realLine; }
        bool rwgtVal::isAll(){ return (idStr == "all"); }
        void rwgtVal::outWrite( REX::paramBlock& srcBlock ){
            if ( isAll() )
            {
                for( auto param : srcBlock.params )
                {
                    param.valStr = valStr;
                    param.modded = true;
                } 
                return;
            } 
            auto currPar = std::find_if( srcBlock.params.begin(), srcBlock.params.end(), 
            [&]( const REX::paramVal& parPar ){ return (parPar.idStr == idStr ); } );
            if( currPar == srcBlock.params.end() ){ 
                srcBlock.params.push_back( REX::paramVal( realLine.substr(realLine.find("set") + 4) ) );
                srcBlock.params[ srcBlock.params.size() - 1 ].modded = true; 
                srcBlock.modded = true;
                return;
            }
            currPar->valStr = valStr;
            currPar->modded = true;
            srcBlock.modded = true;
            return;
        }

        rwgtBlock::rwgtBlock( std::vector<std::string_view> values, std::string_view title)
        {
            name = title;
            rwgtVals.resize( values.size() );
            for( size_t k = 0 ; k < values.size() ; ++k )
            {
                rwgtVals[k] = rwgtVal( values[k] );
            }
        }
        rwgtBlock::rwgtBlock( const std::vector<rwgtVal>& vals, std::string_view title )
        {
            name = title;
            rwgtVals = vals;
        }
        std::string_view rwgtBlock::getBlock(){
            if( written ){ return runBlock; }
            runBlock = "";
            for( auto val : rwgtVals ){
                runBlock += std::string(val.getLine()) + "\n";
            }
            written = true;
            return runBlock;
        }
        void rwgtBlock::outWrite( REX::paramBlock& srcBlock, const std::map<std::string_view, int>& blocks )
        {
            for( auto parm : rwgtVals )
            {
                parm.outWrite( srcBlock );
            }
            srcBlock.modded = true;
            return;
        }

        void rwgtProc::parse(){
            std::vector<std::string_view> blocks;
            std::vector<std::shared_ptr<std::vector<rwgtVal>>> params;
            auto procLines = REX::lineSplitter( procString );
            for( auto line : *procLines )
            {
                if( line.find_first_not_of(" \n\r\f\t\v") == '#' ){ continue; }
                auto strtPt = line.find("set");
                if( strtPt == REX::npos ){ continue; }
                auto words = REX::blankSplitter( line.substr(strtPt) );
                auto currBlock = words->at(1); 
                auto loc = std::find_if( blocks.begin(), blocks.end(), 
                [&]( std::string_view block ){ return (block == currBlock); } );
                if( loc == blocks.end() ){ 
                    blocks.push_back( currBlock ); 
                    params.push_back( std::make_shared<std::vector<rwgtVal>>( std::vector<rwgtVal>({rwgtVal( line )} ) )); }
                else { 
                    params[ std::distance( blocks.begin(), loc )  ]->push_back( rwgtVal( line ) );
                }
            }
            rwgtParams.reserve(blocks.size());
            for( size_t k = 0 ; k < blocks.size() ; ++k )
            {
                rwgtParams.push_back( rwgtBlock( *params[k], blocks[k] ) );
            }
        }
        rwgtProc::rwgtProc( REX::lesHouchesCard slhaSet, std::string_view rwgtSet, bool parseOnline )
        {
            if( rwgtSet == "" ){ return; }
            auto strtLi = rwgtSet.find_first_not_of( " \n\r\f\n\v" );
            if( strtLi == REX::npos ){ return; }
            auto launchPos = rwgtSet.find("launch", strtLi + 1);
            auto commLinePos = rwgtSet.find("#*", strtLi + 1);
            procString = rwgtSet.substr( strtLi, std::min(launchPos, commLinePos) - strtLi );
            if( parseOnline ){ parse(); }
        }
        std::shared_ptr<REX::lesHouchesCard> rwgtProc::outWrite( const REX::lesHouchesCard& paramOrig ){
            auto slhaOrig = std::make_shared<REX::lesHouchesCard>( paramOrig );
            std::map<std::string_view, int> blockIds;
            for( size_t k = 0 ; k < slhaOrig->blocks.size() ; ++k )
            {   slhaOrig->blocks[k].parse( true );
                auto nyama = std::pair<std::string_view, int>( slhaOrig->blocks[k].name, k);
                blockIds.insert( nyama ); }
            for( auto rwgts : rwgtParams )
            { rwgts.outWrite( slhaOrig->blocks[ blockIds.at( rwgts.name ) ], blockIds ); }
            slhaOrig->modded = true;
            return slhaOrig;
        }
        std::string_view rwgtProc::comRunProc(){ return procString; }

        void rwgtCard::parse( bool parseOnline ){
            auto allLaunchPos = REX::findEach( this->srcCard, "launch" );
            std::vector<size_t> lnchPos;
            lnchPos.reserve( allLaunchPos->size() );
            for( auto pos : *allLaunchPos )
            {
                if( pos == 0 ){ lnchPos.push_back(pos); continue; }
                if( srcCard.find_last_of("#", pos) < srcCard.find_last_of("\n", pos) ){ lnchPos.push_back(pos); }
            }
            lnchPos.push_back( REX::npos );
            auto preamble = REX::lineSplitter( srcCard.substr( 0, lnchPos[0] - 1 ) );
            for( auto line : *preamble )
            {
                if( line[line.find_first_not_of(" \n\r\f\t\v")] == '#' ){ continue; }
                opts.push_back( line );
            }
            rwgtNames = std::make_shared<std::vector<std::string>>();
            rwgtNames->reserve( lnchPos.size() - 1 );
            for( size_t k = 0 ; k < lnchPos.size() - 1 ; ++k ){
                auto setPos = srcCard.find( "set", lnchPos[k] );
                if( setPos == REX::npos ){ continue; }
                rwgtRuns.push_back( rwgtProc( slhaCard, srcCard.substr( setPos, lnchPos[k+1] - setPos ), parseOnline ) );
                auto possNamePos = srcCard.find_first_of( "-\n#", lnchPos[k] );
                if( srcCard[possNamePos] == '-' ){
                    auto endLine = srcCard.find( "\n", possNamePos );
                    auto locOpts = srcCard.substr( possNamePos, endLine - possNamePos );
                    rwgtRuns[ rwgtRuns.size() - 1 ].rwgtOpts.push_back( locOpts );
                    auto namePos = locOpts.find( "rwgt_name" );
                    if( namePos != REX::npos ){
                        auto endName = locOpts.find_first_of( " \n\r\f\t\v", namePos );
                        rwgtNames->push_back( std::string( locOpts.substr( namePos + 10, endName - namePos - 10 ) ) );
                    } else {
                        rwgtNames->push_back( "rwgt_" + std::to_string( k + 1 ) );
                    }
                } else {
                    rwgtNames->push_back( "rwgt_" + std::to_string( k + 1 ) );
                }
                rwgtRuns[ rwgtRuns.size() - 1 ].rwgtName = rwgtNames->at( rwgtNames->size() - 1 );
            }
            rwgtProcs = std::vector<std::string_view>(); rwgtProcs.reserve( rwgtRuns.size() );
            for( auto run : rwgtRuns ){
                rwgtProcs.push_back( run.comRunProc() );
            }
        }
        rwgtCard::rwgtCard( std::string_view reweight_card ){
            srcCard = reweight_card;
        }
        rwgtCard::rwgtCard( std::string_view reweight_card, REX::lesHouchesCard slhaParams, bool parseOnline ){
            srcCard = reweight_card;
            slhaCard = slhaParams;
            if( parseOnline ){ parse( parseOnline ); }
        }
        std::vector<std::shared_ptr<REX::lesHouchesCard>> rwgtCard::writeCards( REX::lesHouchesCard& slhaOrig ){
            std::vector<std::shared_ptr<REX::lesHouchesCard>> cardVec;
            slhaOrig.parse();
            cardVec.reserve( rwgtRuns.size() );
            for( auto rwgt : rwgtRuns )
            {
                cardVec.push_back( rwgt.outWrite( slhaOrig ) );
            }
            return cardVec;
        }

        void rwgtCollection::setRwgt( std::shared_ptr<rwgtCard> rwgts ){ 
            if( rwgtSet ){ return; }
            rwgtSets = rwgts; 
            rwgtSet = true;
        }
        void rwgtCollection::setRwgt( rwgtCard rwgts ){ 
            if( rwgtSet ){ return; }
            setRwgt( std::make_shared<rwgtCard>( rwgts ) ); rwgtSet = true; 
        }
        void rwgtCollection::setSlha( std::shared_ptr<REX::lesHouchesCard> slha ){ 
            if( slhaSet ){ return; }
            slhaParameters = slha; 
            slhaParameters->parse(); 
            slhaSet = true; 
        }
        void rwgtCollection::setSlha( REX::lesHouchesCard slha ){ 
            if( slhaSet ){ return; }
            setSlha( std::make_shared<REX::lesHouchesCard>( slha ) ); 
            slhaSet = true;
        }
        void rwgtCollection::setLhe( std::shared_ptr<REX::lheNode> lhe ){ 
            if( lheFileSet ){ return; }
            lheFile = lhe;
            lheFileSet = true;
        }
        void rwgtCollection::setLhe( REX::lheNode& lhe ){ 
            if( lheFileSet ){ return; } 
            setLhe( std::make_shared<REX::lheNode>( lhe ) ); 
            lheFileSet = true;
        }
        void rwgtCollection::setLhe( std::string_view lhe_file ){
            if( lheFileSet ){ return; } 
            lheFile = std::make_shared<REX::lheNode>( REX::lheNode(lhe_file) );
            lheFileSet = true; 
        }
        std::shared_ptr<rwgtCard> rwgtCollection::getRwgt(){ return rwgtSets; }
        std::shared_ptr<REX::lesHouchesCard> rwgtCollection::getSlha(){ return slhaParameters; }
        std::shared_ptr<REX::lheNode> rwgtCollection::getLhe(){ return lheFile; }
        rwgtCollection::rwgtCollection(){ return; }
        rwgtCollection::rwgtCollection( std::shared_ptr<REX::lheNode> lhe, std::shared_ptr<REX::lesHouchesCard> slha, std::shared_ptr<rwgtCard> rwgts ){
            setLhe( lhe );
            setSlha( slha );
            setRwgt( rwgts );
        }
        rwgtCollection::rwgtCollection( const rwgtCollection& rwgts ){
            rwgtSets = rwgts.rwgtSets;
            slhaParameters = rwgts.slhaParameters;
            lheFile = rwgts.lheFile;
            wgts = rwgts.wgts;
            gS = rwgts.gS;
            momenta = rwgts.momenta;
            lheFileSet = rwgts.lheFileSet;
            slhaSet = rwgts.slhaSet;
            rwgtSet = rwgts.rwgtSet;
            skeleton = rwgts.skeleton;
            eventFile = rwgts.eventFile;
            flatWgts = rwgts.flatWgts;
        }
        REX::transSkel& rwgtCollection::getSkeleton(){
            if( !this->skeleton )
                throw std::runtime_error( "Skeleton has not been set." );
            return this->lheSkeleton;
        }
        REX::transSkel& rwgtCollection::getSkeleton( std::vector<REX::eventSet>& evSets ){
            if( this->skeleton ){ return this->lheSkeleton; }
            setSkeleton( evSets );
            return this->lheSkeleton;
        }
        template<class... Args>
        void rwgtCollection::setDoubles(Args&&... args){
            if( lheFile == nullptr || rwgtSets == nullptr || slhaParameters == nullptr )
                throw std::runtime_error( "One or more of the necessary files (SLHA parameter card, LHE event storage file, and MadGraph-format reweight card) have not been initialised." );
            if( this->doublesSet ){ return; }
            if( this->skeleton ){
                this->setDoublesFromSkeleton();
                return;
            }
            REX::lheRetDs returnBools; returnBools.xwgtup = true; returnBools.aqcdup = true; returnBools.pup = true;
            eventFile = REX::transLHE( *lheFile, args... );
            auto vecOfVecs = REX::lheValDoubles( eventFile, returnBools );
            if( vecOfVecs->size() != 3 * eventFile.subProcs.size() )
                throw std::runtime_error( "Incorrect number of parameters have been extracted from the LHE file." );
            for( size_t k = 0 ; k < eventFile.subProcs.size() ; ++k )
            {
                wgts.push_back( vecOfVecs->at( 3*k ) ); 
                gS.push_back( vecOfVecs->at( 3*k + 1 ) ); 
                momenta.push_back( vecOfVecs->at( 3*k + 2 ) );
            }
            flatWgts = eventFile.vectorFlat( wgts );
            this->doublesSet = true;
        }
        void rwgtCollection::setSkeleton( std::vector<REX::eventSet>& evSets ){
            if( lheFile == nullptr || rwgtSets == nullptr || slhaParameters == nullptr )
                throw std::runtime_error( "One or more of the necessary files (SLHA parameter card, LHE event storage file, and MadGraph-format reweight card) have not been initialised." );
            this->lheSkeleton = REX::transSkel( this->lheFile, evSets );
            this->skeleton = true;
        }
        void rwgtCollection::setDoublesFromSkeleton(){
            if( !this->skeleton )
                throw std::runtime_error( "Skeleton has not been set." );
            if( this->doublesSet ){ return; }
            REX::lheRetDs returnBools; returnBools.xwgtup = true; returnBools.aqcdup = true; returnBools.pup = true;
            this->eventFile = REX::transLHE( this->lheSkeleton );
            auto vecOfVecs = REX::lheValDoubles( eventFile, returnBools );
            if( vecOfVecs->size() != 3 * eventFile.subProcs.size() )
                throw std::runtime_error( "Incorrect number of parameters have been extracted from the LHE file." );
            for( size_t k = 0 ; k < eventFile.subProcs.size() ; ++k )
            {
                wgts.push_back( vecOfVecs->at( 3*k ) ); 
                gS.push_back( vecOfVecs->at( 3*k + 1 ) ); 
                momenta.push_back( vecOfVecs->at( 3*k + 2 ) );
            }
            flatWgts = eventFile.vectorFlat( wgts );
            this->doublesSet = true;
        }
        std::shared_ptr<std::vector<std::string>> rwgtCollection::getNames(){ return rwgtSets->rwgtNames; }

        bool rwgtFiles::rwgtPulled(){ return (rewgtCard != nullptr); }
        bool rwgtFiles::slhaPulled(){ return (slhaCard != nullptr); }
        bool rwgtFiles::lhePulled(){ return (lheCard != nullptr); }
        void rwgtFiles::setRwgtPath( std::string_view path ){ rwgtPath = path; }
        void rwgtFiles::setSlhaPath( std::string_view path ){ slhaPath = path; }
        void rwgtFiles::setLhePath( std::string_view path ){ lhePath = path; }
        rwgtFiles::rwgtFiles() : rwgtCollection(){ return; }
        rwgtFiles::rwgtFiles( std::string_view lhe_card, std::string_view slha_card, std::string_view reweight_card ) : rwgtCollection(){
            setRwgtPath( reweight_card );
            setSlhaPath( slha_card );
            setLhePath( lhe_card );
        }
        rwgtFiles::rwgtFiles( const rwgtFiles& rwgts ) : rwgtCollection( rwgts ){
            rwgtPath = rwgts.rwgtPath;
            slhaPath = rwgts.slhaPath;
            lhePath = rwgts.lhePath;
            rewgtCard = rwgts.rewgtCard;
            slhaCard = rwgts.slhaCard;
            lheCard = rwgts.lheCard;
            initialised = rwgts.initialised;
        }
        REX::transSkel& rwgtFiles::initCards( std::vector<REX::eventSet>& evSets ){
            if( initialised ){ return getSkeleton( evSets ); }
            if( rwgtPath == "" || slhaPath == "" || lhePath == "" )
                throw std::runtime_error( "Paths to reweight card, parameter card, or LHE file have not been set" );
            this->pullRwgt(); this->pullSlha(); this->pullLhe();
            this->setLhe( *lheCard );
            this->setSlha( std::make_shared<REX::lesHouchesCard>( *slhaCard ) );
            this->setRwgt( std::make_shared<rwgtCard>( *rewgtCard, *slhaParameters, true ) );
            this->initialised = true;
            return this->getSkeleton( evSets );
        }
        template<class... Args>
        void rwgtFiles::initCards(Args&&... args){
            if( initialised ){ return; }
            if( rwgtPath == "" || slhaPath == "" || lhePath == "" )
                throw std::runtime_error( "Paths to reweight card, parameter card, or LHE file have not been set" );
            pullRwgt(); pullSlha(); pullLhe();
            setLhe( *lheCard );
            setSlha( std::make_shared<REX::lesHouchesCard>( *slhaCard ) );
            setRwgt( std::make_shared<rwgtCard>( *rewgtCard, *slhaParameters, true ) );
            setDoubles(args...);
            initialised = true;
        }
        template<class... Args>
        void rwgtFiles::initCards( std::string_view lhe_card, std::string_view slha_card, std::string_view reweight_card, Args&&... args ){
            setLhePath( lhe_card );
            setSlhaPath( slha_card );
            setRwgtPath( reweight_card );
            initCards(args...);
            initialised = true;
        }
        void rwgtFiles::initDoubles(){
            if( !this->skeleton )
                throw std::runtime_error( "Skeleton has not been set." );
            this->setDoublesFromSkeleton();
        }
        void rwgtFiles::pullRwgt(){
            if( this->rwgtPulled() ){ return; }
            rewgtCard = REX::filePuller( rwgtPath );
        }
        void rwgtFiles::pullSlha(){
            if( this->slhaPulled() ){ return; }
            slhaCard = REX::filePuller( slhaPath );
        }
        void rwgtFiles::pullLhe(){
            if( this->lhePulled() ){ return; }
            lheCard = REX::filePuller( lhePath );
        }

        void rwgtRunner::setMeEval( amplitude eval ){ 
            meEval = eval; meInit = true;
        }
        void rwgtRunner::addMeEval( const REX::event& ev, const amplitude& eval ){}// meEvals.insert( std::pair<REX::event, amplitude>( ev, eval ) ); meCompInit = true; }
        rwgtRunner::rwgtRunner() : rwgtFiles(){ return; }
        rwgtRunner::rwgtRunner( rwgtFiles& rwgts ) : rwgtFiles( rwgts ){ return; }
        rwgtRunner::rwgtRunner( rwgtFiles& rwgts, amplitude meCalc ) : rwgtFiles( rwgts ){
            meEval = meCalc;
            meInit = true;
        }
        rwgtRunner::rwgtRunner( rwgtFiles& rwgts, std::vector<amplitude>& meCalcs ) : rwgtFiles( rwgts ){
            meVec = meCalcs;
            meCompInit = true;
        }
        rwgtRunner::rwgtRunner( std::string_view lhe_card, std::string_view slha_card, std::string_view reweight_card,
        amplitude meCalc ) : rwgtFiles( lhe_card, slha_card, reweight_card ){
            meEval = meCalc;
            meInit = true;
        }
        rwgtRunner::rwgtRunner( const rwgtRunner& rwgts ) : rwgtFiles( rwgts ){
            this->meInit = rwgts.meInit;
            this->meCompInit = rwgts.meCompInit;
            this->meSet = rwgts.meSet;
            this->normWgtSet = rwgts.normWgtSet;
            this->meEval = rwgts.meEval;
            this->meVec = rwgts.meVec;
            this->initMEs = rwgts.initMEs;
            this->meNormWgts = rwgts.meNormWgts;
            this->normWgt = rwgts.normWgt;
            this->rwgtGroup = rwgts.rwgtGroup;
            this->normXSecs = rwgts.normXSecs;
            this->errXSecs = rwgts.errXSecs;
            this->ampNorm = rwgts.ampNorm;
            this->reWgts = rwgts.reWgts;
        }
        bool rwgtRunner::oneME(){ return (meInit != meCompInit); }
        bool rwgtRunner::singAmp(){ return (meInit && !meCompInit); }
        template<class... Args>
        void rwgtRunner::setMEs(Args&&... args){
            initCards(args...);
            normXSecs = std::make_shared<std::vector<double>>( );
            errXSecs = std::make_shared<std::vector<double>>( );
            if( !oneME() )
                throw std::runtime_error( "No or multiple function(s) for evaluating scattering amplitudes has been provided." );
            //ZW FIX THIS
            initMEs = {};
            if( meVec.size() != 0 ){
                for( size_t k = 0 ; k < eventFile.subProcs.size() ; ++k )
                {
                    auto ins = meVec[k]( *(momenta[k]), *(gS[k]) );
                    initMEs.push_back( std::make_shared<std::vector<double>>( ins->begin(), ins->begin() + wgts[k]->size() ) );
                }
            }
            else{
                // DO NOT ALLOW FOR SINGLE ME WITHOUT PASSING EVERYTHING THROUGH VECTOR
            }
            meSet = true;
        }
        bool rwgtRunner::setParamCard( std::shared_ptr<REX::lesHouchesCard> slhaParams ){
            if( slhaPath == "" )
                throw std::runtime_error( "No parameter card path has been provided." );
            if( slhaParameters == nullptr )
                throw std::runtime_error( "No SLHA parameter card has been provided." );
            if( !REX::filePusher( slhaPath, *slhaParams->selfWrite() ) )
                throw std::runtime_error( "Failed to overwrite parameter card." );
            return true;
        }
        void rwgtRunner::setNormWgtsSingleME(){
            meNormWgts = {std::make_shared<std::vector<double>>( wgts[0]->size() )};
            for( size_t k = 0; k < initMEs[0]->size(); k++ ){
                meNormWgts[0]->at( k ) = wgts[0]->at( k ) / initMEs[0]->at( k );
            }
            normWgt = meNormWgts[0];
        }
        void rwgtRunner::setNormWgtsMultiME(){
            meNormWgts = std::vector<std::shared_ptr<std::vector<double>>>( initMEs.size() );
            for( size_t k = 0 ; k < wgts.size() ; ++k ){
                meNormWgts[k] = std::make_shared<std::vector<double>>( wgts[k]->size() );
                for( size_t i = 0 ; i < wgts[k]->size() ; ++i ){
                    meNormWgts[k]->at( i ) = wgts[k]->at( i ) / initMEs[k]->at( i );
                }
            }
            normWgt = eventFile.vectorFlat( meNormWgts );
        }
        void rwgtRunner::setAmpNorm( double precision ){
            if( this->ampNorm != 0.0 ){ return; }
            auto xSecLines = this->lheFile->getInit()->getLines();
            if( xSecLines.size() > 1 ){
                std::cout << "\n\033[1;33mWarning: Multiple cross-section lines found in LHE file.\nAssuming total cross section given by sum of all cross sections.\033[0m\n";
            }
            if( xSecLines.size() == 0 )
                throw std::runtime_error( "No cross-section information found in LHE file." );
            double xSec = 0.0;
            for( size_t k = 0 ; k < xSecLines.size() ; ++k ){
                xSec += std::stod(std::string(xSecLines[k]->xsecup));
            }
            double div = 0.0;
            bool sameWeight = true;
            for( size_t k = 1 ; k < this->flatWgts->size() - 1 ; k += size_t(flatWgts->size()/21) ){
                if( std::abs( flatWgts->at(0) - flatWgts->at(0) ) > precision ){
                    sameWeight = false;
                    break;
                }
            }
            if( sameWeight ){
                if( std::abs(xSec - flatWgts->at(0)) < precision ){
                    this->ampNorm =  1. / double(flatWgts->size());
                    return;
                }
                div = flatWgts->size() * flatWgts->at(0);
            }
            else{
                div = std::accumulate( flatWgts->begin(), flatWgts->end(), 0.0 );
            }
            this->ampNorm = xSec / div;
        }
        template<class... Args>
        void rwgtRunner::setNormWgts(Args&&... args){
            if( !oneME() ){ setMEs(args...); } 
            for( size_t k = 0; k < initMEs.size() ; ++k ){
                if( initMEs[k]->size() != wgts[k]->size() )
                    throw std::runtime_error( "Inconsistent number of events and event weights." );
            }
            if( initMEs.size() == 1 ){ setNormWgtsSingleME(); }
            else { setNormWgtsMultiME(); }
            normWgtSet = true;
        }
        bool rwgtRunner::singleRwgtIter( std::shared_ptr<REX::lesHouchesCard> slhaParams, std::shared_ptr<REX::lheNode> lheIn, size_t currId ){
            if( !normWgtSet )
                throw std::runtime_error( "Normalised original weights (wgt/|ME|) not evaluated -- new weights cannot be calculated." );
            if( !setParamCard( slhaParams ) )
                throw std::runtime_error( "Failed to rewrite parameter card." );
            std::shared_ptr<std::vector<double>> newWGTs;
            if( singAmp() ){
                auto newMEs = meEval( *momenta[0], *gS[0] );
                newWGTs = REX::vecElemMult( *newMEs, *meNormWgts[0] );
            }
            else{
                std::vector<std::shared_ptr<std::vector<double>>> nuMEs = {};
                std::shared_ptr<std::vector<double>> newMEs = eventFile.vectorFlat( nuMEs );
                newWGTs = REX::vecElemMult( *newMEs, *normWgt );
            }
            //ZW IF MULTIPLE TYPES
            reWgts->push_back( newWGTs );
            REX::newWgt nuWgt( rwgtSets->rwgtRuns[currId].comRunProc(), reWgts->at(reWgts->size() - 1) );
            lheIn->addWgt( 0, nuWgt );
            return true;
        }
        bool rwgtRunner::singleRwgtIter( std::shared_ptr<REX::lesHouchesCard> slhaParams, std::shared_ptr<REX::lheNode> lheIn, size_t currId, std::string& id ){
            if( !normWgtSet )
                throw std::runtime_error( "Normalised original weights (wgt/|ME|) not evaluated -- new weights cannot be calculated." );
            if( !setParamCard( slhaParams ) )
                throw std::runtime_error( "Failed to rewrite parameter card." );
            std::shared_ptr<std::vector<double>> newWGTs;
            if( singAmp() ){
                auto newMEs = meEval( *momenta[0], *gS[0] );
                newWGTs = REX::vecElemMult( *newMEs, *meNormWgts[0] );
            }
            else{
                std::vector<std::shared_ptr<std::vector<double>>> nuMEs = {};
                for( size_t k = 0 ; k < eventFile.subProcs.size() ; ++k )
                {
                    nuMEs.push_back(meVec[k]( *(momenta[k]), *(gS[k]) ));
                }
                std::shared_ptr<std::vector<double>> newMEs = eventFile.vectorFlat( nuMEs );
                newWGTs = REX::vecElemMult( *newMEs, *normWgt );
            }
            //ZW IF MULTIPLE TYPES
            reWgts->push_back( newWGTs );
            REX::newWgt nuWgt( rwgtSets->rwgtRuns[currId].comRunProc(), reWgts->at(reWgts->size() - 1), id );
            lheIn->addWgt( 0, nuWgt );
            return true;
        }
        bool rwgtRunner::singleRwgtIter( std::shared_ptr<REX::lesHouchesCard> slhaParams, std::shared_ptr<REX::lheNode> lheIn, size_t currId, REX::event& ev ){
            if( !normWgtSet )
                throw std::runtime_error( "Normalised original weights (wgt/|ME|) not evaluated -- new weights cannot be calculated." );
            if( !setParamCard( slhaParams ) )
                throw std::runtime_error( "Failed to rewrite parameter card." );
            std::shared_ptr<std::vector<double>> newWGTs;
            if( singAmp() ){
                auto newMEs = meEval( *momenta[0], *gS[0] );
                newWGTs = REX::vecElemMult( *newMEs, *meNormWgts[0] );
            }
            else{
                std::vector<std::shared_ptr<std::vector<double>>> nuMEs = {};
                std::shared_ptr<std::vector<double>> newMEs = eventFile.vectorFlat( nuMEs );
                newWGTs = REX::vecElemMult( *newMEs, *normWgt );
            }
            //ZW IF MULTIPLE TYPES
            reWgts->push_back( newWGTs );
            REX::newWgt nuWgt( rwgtSets->rwgtRuns[currId].comRunProc(), reWgts->at(reWgts->size() - 1) );
            lheIn->addWgt( 0, nuWgt );
            return true;
        }
        bool rwgtRunner::singleRwgtIter( std::shared_ptr<REX::lesHouchesCard> slhaParams, std::shared_ptr<REX::lheNode> lheIn, size_t currId, 
        std::string& id, REX::event& ev ){
            if( !normWgtSet )
                throw std::runtime_error( "Normalised original weights (wgt/|ME|) not evaluated -- new weights cannot be calculated." );
            if( !setParamCard( slhaParams ) )
                throw std::runtime_error( "Failed to rewrite parameter card." );
            std::shared_ptr<std::vector<double>> newWGTs;
            if( singAmp() ){
                auto newMEs = meEval( *momenta[0], *gS[0] );
                newWGTs = REX::vecElemMult( *newMEs, *meNormWgts[0] );
            }
            else{
                std::vector<std::shared_ptr<std::vector<double>>> nuMEs = {};
                std::shared_ptr<std::vector<double>> newMEs = eventFile.vectorFlat( nuMEs );
                newWGTs = REX::vecElemMult( *newMEs, *normWgt );
            }
            //ZW IF MULTIPLE TYPES
            reWgts->push_back( newWGTs );
            REX::newWgt nuWgt( rwgtSets->rwgtRuns[currId].comRunProc(), reWgts->at(reWgts->size() - 1), id );
            lheIn->addWgt( 0, nuWgt );
            return true;
        }
        bool rwgtRunner::lheFileWriter( std::shared_ptr<REX::lheNode> lheIn, std::string outputDir ){
            bool writeSuccess = REX::filePusher( outputDir, *lheIn->nodeWriter() );
            if( !writeSuccess )
                throw std::runtime_error( "Failed to write LHE file." );
            return true;
        }
        bool rwgtRunner::calcXSecs(){
            if( normXSecs->size() != 0 ){ return true; }
            if( ampNorm == 0.0 )
                throw std::runtime_error( "Normalisation factor for scattering amplitudes has not been calculated.\nReweighted LHE file has been written, but may contain errors." );
            if( reWgts->size() == 0 )
                throw std::runtime_error( "No reweighting has been performed, or new weights have not been stored properly.\nReweighted LHE file has been written, but may contain errors." );
            for( size_t k = 0 ; k < reWgts->size() ; ++k ){
                normXSecs->push_back( ampNorm * std::accumulate( reWgts->at(k)->begin(), reWgts->at(k)->end(), 0.0 ) );
            }
            return true;
        }
        bool rwgtRunner::calcXErrs(){
            if( errXSecs->size() != 0 ){ return true; }
            if( reWgts->size() == 0 )
                throw std::runtime_error( "No reweighting has been performed, or new weights have not been stored properly.\nReweighted LHE file has been written, but may contain errors." );
            if( normXSecs->size() != reWgts->size() )
                throw std::runtime_error( "Different number of reweighted event sets and reweighted cross sections internally.\nReweighted LHE file has been written, but may contain errors." );
            double invN = 1. / double(reWgts->at(0)->size());
            double sqrtInvN = std::sqrt( invN );
            auto xSecLines = this->lheFile->getInit()->getLines();
            double xSec = 0.0;
            double xErr = 0.0;
            for( size_t k = 0 ; k < xSecLines.size() ; ++k ){
                xSec += std::stod(std::string(xSecLines[k]->xsecup));
                xErr += std::pow(std::stod(std::string(xSecLines[k]->xerrup)),2);
            }
            xErr = std::sqrt( xErr );
            for( size_t k = 0 ; k < reWgts->size() ; ++k ){
                double xSecCurr = normXSecs->at(k);
                auto locWgts = reWgts->at(k);
                double omega = 0.0;
                double omegaSqr = 0.0;
                for( auto wgt : *locWgts ){
                    double invWgt = 1. / wgt;
                    omega += invWgt;
                    omegaSqr += invWgt * invWgt;
                }
                double var = (omegaSqr - omega * omega * invN) * invN * xSecCurr * xSecCurr;
                double error = std::sqrt( std::max( 0., sqrtInvN * var) )*xSec + xSecCurr * omega * invN * xErr;
                if( std::isnan( error ) || std::isinf( error ) ){
                    std::cout << "\033[1;33mWarning:Error propagation yielded NaN for " << rwgtSets->rwgtNames->at(k) << ". Approximating the error at cross section level.\033[0m\n";
                    error = xErr * std::max( xSec / xSecCurr, xSecCurr / xSec );
                }
                errXSecs->push_back( error ); 
            }
            return true;
        }
        void rwgtRunner::runRwgt( const std::string& output, double precision ){
            reWgts = std::make_shared<std::vector<std::shared_ptr<std::vector<double>>>>( std::vector<std::shared_ptr<std::vector<double>>>() );
            setAmpNorm( precision );
            setMEs();
            setNormWgts(); 
            rwgtGroup = std::make_shared<REX::weightGroup>(); 
            auto currInd = lheFile->getHeader()->addWgtGroup( rwgtGroup ); 
            auto paramSets = rwgtSets->writeCards( *slhaParameters ); 
            for( size_t k = 0 ; k < paramSets.size(); k++ ){ 
                singleRwgtIter( paramSets[k], lheFile, k, rwgtSets->rwgtNames->at(k) ); 
                std::cout << ".";
            }
            lheFileWriter( lheFile, output ); 
            REX::filePusher( slhaPath, *slhaCard ); 
            std::cout << "\nReweighting done.\n";
        }
        std::shared_ptr<std::vector<double>> rwgtRunner::getReXSecs(){
            if(this->calcXSecs()){ return normXSecs; }
            return nullptr;
        }
        std::shared_ptr<std::vector<double>> rwgtRunner::getReXErrs(){
            if(this->calcXErrs()){ return errXSecs; }
            return nullptr;
        }

}

#endif
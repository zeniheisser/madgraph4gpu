// ZW: header for LHEF parsing
// uses boost (rapidXML) for LHEF parsing
#include <iostream>
#include <string>
#include <set>
#include <cmath>
#include <utility>

#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>

// ZW: only use the property_tree (and associated rapidXML implementation)
// portions of boost, so shorten namespace for simplicity
namespace pt = boost::property_tree;
// ZW: all fcns within the PEP standard sit in the
// namespace PEP
namespace PEP
{
    // ZW: namespace for previous iteration of PEP
    // which relied on an assumed structure of the LHE
    // based on spacing
    // New iteration only assumes the explicit LHE standard
    // and using these functions, although faster,
    // may parse files incorrectly
    namespace STRINGREAD{

// ZW: function for extracting the necessary information
// to regenerate the matrix elements of an LHE file
// assuming the process is already known (future development)
// Returns an std::vector of doubles, where the first
// 4*noParticles*noEvents are the momenta of external particles,
// (ordered as vector[4*noParticles*currentEvent + 4*currentParticle + currentMomentumComp])
// the next noEvents elements are the alphaS ordered by event,
// and the last two elements are the number of particles and number of events repsectively
std::vector<double>& eventExtraction ( std::string fileName ) {

    pt::ptree eventFile;

    try {
        pt::read_xml(fileName, eventFile);
    } catch (pt::xml_parser_error &e) {
        std :: cout << "Failed to parse LHE file" << e.what();
    } catch (...) {        std :: cout << "Undefined error while parsing LHE file";
    }

    // ZW: get the generator
    // development used MG5aMC so could make simplifications for that,
    // but the extraction works for general LHE files 
    auto genName = eventFile.get<std::string>("LesHouchesEvents.init.generator.<xmlattr>.name");

    // ZW: extract number of events, either using the generation info (if MG5aMC)
    // or by just counting the number of event tags (other generators)
    int noEvents = 0;
    if (genName == "MadGraph5_aMC@NLO") {
        auto& mgInfo = eventFile.get_child("LesHouchesEvents.header.MGGenerationInfo");
        auto lineEnd = mgInfo.data().substr(2,60).find("\n");
        noEvents = std::stoi(mgInfo.data().substr(32, lineEnd + 2));
    } else {
        for (auto& event : eventFile.get_child("LesHouchesEvents")) {
            if (event.first != "event"){
                continue;
            }
            noEvents += 1;
        }
    }

    // ZW: extract no of particles from first event
    auto& firstEv = eventFile.get_child("LesHouchesEvents.event");
    int noPrts = std::stoi(firstEv.data().substr(0,7));
    // ZW: calculate number of elems necessary, 4 for each particle for each event,
    // plus one for each event for the alphaS
    int noElems = noEvents*(noPrts*4 + 1 + 1);
    // ZW: set up output vector with two more elements than noElems,
    // so we can also return noPrts and noElems
    static std::vector<double> momentumVec(noElems + 2);
    momentumVec[noElems] = noPrts;
    momentumVec[noElems + 1] = noEvents;

    // ZW: looping over children nodes of LHE file, but need to
    // keep track of event ordering, so we create a dummy loop
    // variable to remember current event number
    int currEv = 0;
    for (auto& event : eventFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        // ZW: for each new event, find the linebreak from the first line (event information)
        // where it switches to the second line (first real particle line)
        auto startPos = event.second.data().find("\n", 8); 
        // ZW: append alphaS of current event
        // ZW: update -- for now append g_S instead of alphaS
        momentumVec[noEvents*noPrts*4 + currEv] = std::sqrt( 4.0 * M_PI * std::stod(event.second.data().substr(startPos - 15, 16)));
        momentumVec[noEvents*(noPrts*4 + 1) + currEv] = std::stod(event.second.data().substr(startPos - 60, 16));
        // ZW: loop over all particles in current event
        for (int currPrt = 0; currPrt < noPrts; currPrt++ ) {
            // ZW: loop over each momentum component of current particle
            for (int currMom = 0; currMom < 4; currMom++){
                // ZW: parse the particle line, extracting the current momentum and appending it to vector
                momentumVec[currEv*noPrts*4 + currPrt*4 + currMom] =  std::stod(event.second.data().substr(startPos + 34 + currMom*18));
            }
            // ZW: update startPos so that the next particle iteration will start at the break between current particle and next
            startPos = event.second.data().find("\n", startPos + 2);
        }
        currEv += 1;
    }

    return momentumVec;
}

std::vector<std::string>& procList ( pt::ptree &eventFile ) {

    // ZW: get the generator
    // development used MG5aMC so could make simplifications for that,
    // but the extraction works for general LHE files 
    auto genName = eventFile.get<std::string>("LesHouchesEvents.init.generator.<xmlattr>.name");

    // ZW: extract number of events, either using the generation info (if MG5aMC)
    // or by just counting the number of event tags (other generators)
    int noEvents = 0;
    if (genName == "MadGraph5_aMC@NLO") {
        auto& mgInfo = eventFile.get_child("LesHouchesEvents.header.MGGenerationInfo");
        auto lineEnd = mgInfo.data().substr(2,60).find("\n");
        noEvents = std::stoi(mgInfo.data().substr(32, lineEnd + 2));
    } else {
        for (auto& event : eventFile.get_child("LesHouchesEvents")) {
            if (event.first != "event"){
                continue;
            }
            noEvents += 1;
        }
    }

    static std::vector<std::string> procsList(noEvents);

    // ZW: looping over children nodes of LHE file, but need to
    // keep track of event ordering, so we create a dummy loop
    // variable to remember current event number
    int currEv = 0;
    for (auto& event : eventFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        // ZW: for each new event, find the linebreak from the first line (event information)
        // where it switches to the second line (first real particle line)
        auto startPos = event.second.data().find("\n", 8); 
        auto noPrts = std::stoi(event.second.data().substr(0,7));
        std::string thisLine = "";
        bool toPtNFnd = true;
        auto prtState = std::stoi(event.second.data().substr(startPos + 11));
        // ZW: loop over all particles in current event
        for (int currPrt = 0; currPrt < noPrts; currPrt++ ) {
            // ZW: loop over each momentum component of current particle
            thisLine += std::to_string(std::stoi(event.second.data().substr(startPos)));
            thisLine += " ";
            // ZW: update startPos so that the next particle iteration will start at the break between current particle and next
            startPos = event.second.data().find("\n", startPos + 2);
            if( toPtNFnd ){
                if( std::stoi(event.second.data().substr(startPos + 11)) != prtState ) {
                    thisLine += " > ";
                    toPtNFnd = false;
                }
            }
        }
        procsList[currEv] = thisLine;
        currEv += 1;
    }
    return procsList;
}

std::set<std::pair<std::string, int>>& procExtractor ( pt::ptree &eventFile ) {

    // ZW: get the generator
    // development used MG5aMC so could make simplifications for that,
    // but the extraction works for general LHE files 
    auto genName = eventFile.get<std::string>("LesHouchesEvents.init.generator.<xmlattr>.name");

    static std::set<std::pair<std::string, int>> procSet;

    // ZW: looping over children nodes of LHE file, but need to
    // keep track of event ordering, so we create a dummy loop
    // variable to remember current event number
    for (auto& event : eventFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        // ZW: for each new event, find the linebreak from the first line (event information)
        // where it switches to the second line (first real particle line)
        auto startPos = event.second.data().find("\n", 8); 
        auto noPrts = std::stoi(event.second.data().substr(0,7));
        std::string thisLine = "";
        bool toPtNFnd = true;
        auto prtState = std::stoi(event.second.data().substr(startPos + 11));
        // ZW: loop over all particles in current event
        for (int currPrt = 0; currPrt < noPrts; currPrt++ ) {
            // ZW: loop over each momentum component of current particle
            thisLine += std::to_string(std::stoi(event.second.data().substr(startPos)));
            thisLine += " ";
            // ZW: update startPos so that the next particle iteration will start at the break between current particle and next
            startPos = event.second.data().find("\n", startPos + 2);
            if( toPtNFnd ){
                if( std::stoi(event.second.data().substr(startPos + 11)) != prtState ) {
                    thisLine += " > ";
                    toPtNFnd = false;
                }
            }
        }
        if (procSet.find( std::make_pair( thisLine, noPrts ) ) == procSet.end()){
            procSet.insert(std::make_pair( thisLine, noPrts ));
        }
    }
    return procSet;
}
    }

// ZW: deprecated function, returns every parameter in the final event node
// as a vector of strings
std::vector<std::string>& pepSplitter ( pt::ptree &eventFile ) {

    static std::vector<std::string> procElems;
    static std::vector<std::string> trueElems;

    // ZW: looping over children nodes of LHE file
    for (auto event : eventFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        // ZW: for each new event, replace all linebreaks with blankspace
        // and then split the string into a vector
        std::replace( event.second.data().begin(), event.second.data().end(), '\n', ' ');
        boost::split(procElems, event.second.data(), boost::is_any_of(" "));
        int falseSize = std::count(procElems.begin(), procElems.end(), "");
        trueElems.resize(procElems.size() - falseSize);
        // ZW: explicitly get rid of all the elements containing only the null characters
        // which come from repeated blankspaces in the string
        int trueSize = 0;
        for( int k = 0; k < procElems.size(); ++k){
            if( procElems[k] != ""){
                trueElems[trueSize] = procElems[k];
                trueSize += 1;
            }
        }
    }
    return trueElems;
}

// ZW: deprecated function, returns each mandatory parameter of
// the final event node of the LHE file
// Works more or less like pepSplitter but only returns the event and
// particle information from the event
// (ie no extra information that could be added and still maintain
// the LHE standard)
std::vector<std::string>& eventExtractor( pt::ptree &eventFile ) {
    std::vector<std::string> procElems;
    static std::vector<std::string> trueElems;
    for (auto event : eventFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        std::replace( event.second.data().begin(), event.second.data().end(), '\n', ' ');
        boost::split(procElems, event.second.data(), boost::is_any_of(" "));
        procElems.erase(std::remove(procElems.begin(), procElems.end(), ""));
        const int noPrt = std::stoi(procElems[0]);
        int totNumElems = 6 + 13 * noPrt;
        trueElems.resize(totNumElems);
        for (auto currElem = 0; currElem < totNumElems; ++currElem)
        {
            trueElems[currElem] = procElems[currElem];
        }
    }
    return trueElems;
}

// ZW: function for loading an XML (LHE) file as a propertytree
// to clean up other functions
// Just tries using the boost XML reader to load the LHEF
// and catches the boost errors if it fails
pt::ptree& fileLoader ( std::string fileName ) {
    static pt::ptree eventFile;
    try {
        pt::read_xml(fileName, eventFile);
    } catch (pt::xml_parser_error &e) {
        std :: cout << "Failed to parse LHE file" << e.what();
    } catch (...) {        std :: cout << "Undefined error while parsing LHE file";
    }

    return eventFile;
}

// ZW: returns the total number of particles in the LHE
// NOTE: NOT the number of particles per event NOR
// a vector of particles for each event,
// but the sum of number of particles from each event
int& noPrt( pt::ptree& eventFile ) {
    // ZW: extract number of particles per event
    // by just counting the number prts per event tag
    static int noPrts = 0;
    for (auto& event : eventFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        noPrts += std::stoi(eventFile.get_child("LesHouchesEvents.event").data().substr(0,7));
    }
    return noPrts;
}

// ZW: returns total number of events within the LHEF by
// looping over the number of nodes with the tag "event"
int& noEvt( pt::ptree& eventFile ) {
    static int noEvents = 0;
    for (auto& event : eventFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        noEvents += 1;
    }
    return noEvents;
}

// ZW: all-encompassing function for parsing LHEF
// which contains only one type of process
// Works (relatively quickly, ~2 times slower than pure string parsing)
// but has no safety in case of LHEFs with multiple processes
std::vector<std::vector<double>*>& eventParser( std::string lheFile ) {
    bool getGs = true;
    pt::ptree parseFile = fileLoader( lheFile );
    int noPrts = noPrt( parseFile );
    int noEvts = noEvt( parseFile );
    int noPrtsRemain;
    int nWarpRemain = (32 - ( noEvts % 32 )) % 32;
    if ( noPrts % noEvts == 0){
        noPrtsRemain = 4 * nWarpRemain * int( noPrts / noEvts );
    }   else {
        std::cout << "\nNo. of particles is not a multiple of the no of events -- this function only takes singular processes.\n";
        noPrtsRemain = 4 * 32 * 10;
    }
    static std::vector<double> eventVector( 6*noEvts + 13*noPrts );
    static std::vector<double> momVector( 4*noPrts + noPrtsRemain );
    static std::vector<double> alphaVector( noEvts + nWarpRemain );
    std::vector<std::string> procElems;
    int indexElement = 0;
    int momIndex = 0;
    int alphaIndex = 0;

    for (auto event : parseFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        std::replace( event.second.data().begin(), event.second.data().end(), '\n', ' ');
        boost::split(procElems, event.second.data(), boost::is_any_of(" "));
        procElems.erase(std::remove(procElems.begin(), procElems.end(), ""));
        const int noPrt = std::stoi(procElems[0]);
        int totNumElems = 6 + 13 * noPrt;
        for (auto currElem = 0; currElem < totNumElems; ++currElem)
        {
            eventVector[indexElement] = std::stod(procElems[currElem]);
            indexElement += 1;
        }
        for ( auto prts = 0; prts < noPrt; ++prts )
        {
            momVector[momIndex] = std::stod(procElems[6 + 13*prts + 9]);
            momIndex += 1;
            for ( auto momComp = 0; momComp < 3; ++momComp )
            {
                momVector[momIndex] = std::stod(procElems[6 + 13*prts + 6 + momComp]);
                momIndex += 1;
            }
        }
        if( getGs ){
            alphaVector[alphaIndex] = std::sqrt( 4.0 * M_PI * std::stod(procElems[5]));
        } else {
            alphaVector[alphaIndex] = std::stod(procElems[5]);
        }
        alphaIndex += 1;
    }
    static std::vector<std::vector<double>*> ptrVec{ &eventVector, &momVector, &alphaVector };
    return ptrVec;
}


// ZW: turning string into a vector of strings, split by blankspaces and newlines
// and get rid of any vector entries that are just the null character
std::vector<std::string>& stringSplitter( std::string& currEvent ){
    std::vector<std::string> procElems;
    std::replace( currEvent.begin(), currEvent.end(), '\n', ' ');
    boost::split(procElems, currEvent, boost::is_any_of(" "));
    const int truVal = std::count(procElems.begin(), procElems.end(), "");
    //procElems.erase(std::remove(procElems.begin(), procElems.end(), ""));
    static std::vector<std::string> trueElems( truVal );
    int currVal = 0;
    for( auto line : procElems )
    {
        if( line == "" ){
            continue;
        }
        trueElems[currVal] = line;
        ++currVal;
    }
    return trueElems;
}

// ZW: function for extracting the process from an LHE event block
// in terms of the PDG codes, starting with the number of particles
// (which is not necessarily no of external particles, if propagators
// are specified)
// ie the process g g to t tbar would be "4: 21 21 > 6 -6"
std::string procReader( std::string currEvent ){
    std::vector<std::string> eventElems = stringSplitter( currEvent );
    std::string process = eventElems[0] + ":";
    unsigned int nPrt = std::stoi(eventElems[0]);
    bool prtStatus = true;
    for( unsigned int prtcl = 0; prtcl < nPrt; ++prtcl)
    {
        process +=  " " + eventElems[6 + 13*prtcl];
        if ( prtStatus ){
            if ( eventElems[7 + 13*prtcl] != eventElems[7 + 13*(prtcl+1)]){
                process += " >";
                if( eventElems[7 + 13*(prtcl+1)] == "1" ){
                    prtStatus = false;
                }
            }
        }
    } 
    return process;
}

// ZW: extracts the process ordering within the LHEF, ordered by where the first
// instance of an process occurs in the LHEF andd returns a vector
// of pointers to vectors of bools, where the vectors of bools tracks
// to which events of the LHE corresponds to which process, where
// true means a given event is of the given process
std::vector<std::vector<bool>*>& procOrder( pt::ptree& eventFile, std::vector<std::string> evtSet, unsigned int nEvt ) {
    static std::vector<std::vector<bool>*> eventBools( evtSet.size());
    /* for ( int k = 0 ; k < eventBools.size() ; ++k )
    {
        std::cout << "\n439\n";
        (*eventBools[k]).reserve( nEvt );
        std::cout << "\n441\n";
        std::fill( eventBools[k]->begin(), eventBools[k]->end(), false );
        std::cout << "\n443\n";
    } */

    static std::vector<std::vector<bool>> pracBools( evtSet.size(), std::vector<bool> ( nEvt ));
    for( auto boolSets : pracBools ){
        std::fill( boolSets.begin(), boolSets.end(), false );
    }

    unsigned int currEv = 0;
    for (auto event : eventFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        std::string currProc = procReader( event.second.data() );
        auto corrInd = std::find( evtSet.begin(), evtSet.end(), currProc );
        pracBools[std::distance( evtSet.begin(), corrInd )][currEv] = true;
        //(*corrInd) = true;
        /* for ( unsigned int k = 0; k < evtSet.size(); ++k) {
            if ( currProc == evtSet[k] )
            {
                (*eventBools[k])[currEv] = true;
            } else {
                (*eventBools[k])[currEv] = false;
            }
        } */
        currEv += 1;
    }
    for( int k = 0 ; k < eventBools.size() ; ++k )
    {
        eventBools[k] = &pracBools[k];
    }
    return eventBools;
}
// ZW: extracts which processes occur in an LHEF, ordered by where the first
// instance of an process occurs in the LHEF, ie the first event is of process 1,
// the first event (in order) of a different process is process 2 etc
std::vector<std::string>& processExtractor( pt::ptree& eventFile ) {
    static std::vector<std::string> processes;
    for (auto event : eventFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        std::string currProc = procReader( event.second.data() );
        if ( std::none_of(processes.cbegin(), processes.cend(), [&currProc](std::string proc){ return proc == currProc;}))
        {
            processes.push_back(currProc);
        }
    }
    return processes;
}

// ZW: function for parsing a single type of process from an LHEF
// eventFile is the propertytree of the LHEF
// relEv is a vector of bools which states which of the events to parse (ie which have the considered process)
// nEvt is an integer of the number of event corresponding to the given process
// nPrt is the number of external particles within the given process
std::vector<std::vector<double>*>& singleEventParser( pt::ptree& eventFile, const std::vector<bool>& relEv, const unsigned int nPrt ) {
    // ZW: getGs says whether to return the g_S (true) or the alpha_S(false)
    bool getGs = true;
    // ZW: nuEvt is the total number of relevant events rounded up
    // to the nearest multiple of 32
    int nEvt = std::count( relEv.begin(), relEv.end(), true );
    //int nEvt = 10000; //THIS IS THE FUCKER
    unsigned int nuEvt = nEvt + ((32 - ( nEvt % 32 )) % 32);
    // ZW: momVector is the returned vector of 4-momenta,
    // (currently) ordered as (E, px, py, pz)
    static std::vector<double> momVector( 4 * nuEvt * nPrt);
    // ZW: alphaVector is the returned vector of alphas or gs
    static std::vector<double> alphaVector( nuEvt );
    static std::vector<double> wgtVector( nEvt );
    // ZW: dummy indices to keep track of RELEVANT momenta, alphas, and event
    unsigned int momIndex = 0;
    unsigned int alphaIndex = 0;
    unsigned int wgtIndex = 0;
    unsigned int currEvt = 0;
    for (auto event : eventFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        // ZW: check if event should be considered 
        if (relEv[currEvt] ) {
        // ZW: turning event block into a vector of strings
        if( alphaIndex > nuEvt ){std::cout << "\n\nINDEX TOO DAMN BIG\n\n";}
        if( momIndex > momVector.size() ){std::cout << "\n\nMOMINDEX TOO DAMN BIG\n\n";}
        auto procElems = stringSplitter(event.second.data());
        // ZW: appending the momenta, ordered as (E,px,py,pz)
        for ( auto prts = 0; prts < nPrt; ++prts )
        {
            momVector[momIndex] = std::stod(procElems[6 + 13*prts + 9]);
            momIndex += 1;
            for ( auto momComp = 0; momComp < 3; ++momComp )
            {
                momVector[momIndex] = std::stod(procElems[6 + 13*prts + 6 + momComp]);
                momIndex += 1;
            }
        }
        // ZW: append the alphas or gs
        if( getGs ){
            alphaVector[alphaIndex] = std::sqrt( 4.0 * M_PI * std::stod(procElems[5]));
        } else {
            alphaVector[alphaIndex] = std::stod(procElems[5]);
        }
        alphaIndex += 1;
        //wgtVector[ wgtIndex ] = std::stod( procElems[2] );
        //++wgtIndex;
        }
        currEvt += 1;
    }
    // ZW: declare the vector of pointers to the vectors of momenta and alphas
    static std::vector<std::vector<double>*> ptrVec{ &momVector, &alphaVector, &wgtVector };
    return ptrVec;
}

// ZW: wrapper for parsing LHEFs, without assumption of only containing a single type of process
// Returns vector of pointers to vectors of doubles, where every sequential pair of doubles
// corresponds to one type of process contained within the LHEF, order by their order of appearance
// in the LHEF
std::vector<std::vector<double>*>& multiEventParser( pt::ptree& eventFile ){
    std::vector<std::string> procList = processExtractor( eventFile );
    for( auto procs : procList ){
    }
    std::vector<unsigned int> numPrts(procList.size());
    for ( unsigned int k = 0; k < procList.size(); ++k )
    {
        numPrts[k] = std::stoi(procList[k].substr(0,1));
    }
    static std::vector<std::vector<double>*> vecPtrs( 3 * procList.size() );
    vecPtrs.clear();
    unsigned int nEvt = noEvt( eventFile );
    std::vector<std::vector<bool>*> procOrdering = procOrder( eventFile, procList, nEvt );
    for (unsigned int k = 0; k < procList.size(); ++k )
    {
        std::cout << "\n\nNOT YET CRASHED\n\n";
        auto processVecs = singleEventParser( eventFile, *procOrdering[k], numPrts[k] );
        std::cout << "\n\nSTILL NOT CRASHED\n\n";
        std::cout << "\nnr of rel procs is  " << std::count( procOrdering[k]->begin(), procOrdering[k]->end(), true );
        for( int m = 0 ; m < processVecs.size() ; ++m ){
            std::cout << "\n\nABOUT TO ADD SOME SHIT TO THE VECTOR\n\n";
            vecPtrs[k*processVecs.size() + m] = processVecs[m];
        }
    }
    return vecPtrs;
}

// ZW: wrapper for multiEventParser so user only needs to apply the filename
// of the LHEF
// Should eventually be extended to also work for zipped files
std::vector<std::vector<double>*>& lheParser( std::string fileName ){
    pt::ptree lheFile = fileLoader( fileName );
    return multiEventParser(lheFile);
}

}
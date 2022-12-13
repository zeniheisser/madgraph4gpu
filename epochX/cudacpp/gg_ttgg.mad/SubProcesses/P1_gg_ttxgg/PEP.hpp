// ZW: header for LHEF parsing
// uses boost (rapidXML) for LHE file parsing
// although LHEF does not fulfil the XML standard totally
// it is sufficiently similar for this purpose
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
        //std::cout << "in an event\n";
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

std::vector<std::string>& pepSplitter ( pt::ptree &eventFile ) {

    static std::vector<std::string> procElems;
    static std::vector<std::string> trueElems;

    // ZW: looping over children nodes of LHE file, but need to
    // keep track of event ordering, so we create a dummy loop
    // variable to remember current event number
    for (auto event : eventFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        // ZW: for each new event, find the linebreak from the first line (event information)
        // where it switches to the second line (first real particle line)
        std::replace( event.second.data().begin(), event.second.data().end(), '\n', ' ');
        boost::split(procElems, event.second.data(), boost::is_any_of(" "));
        int falseSize = std::count(procElems.begin(), procElems.end(), "");
        trueElems.resize(procElems.size() - falseSize);
        int trueSize = 0;
        for( int k = 0; k < procElems.size(); ++k){
            if( procElems[k] != ""){
                trueElems[trueSize] = procElems[k];
                trueSize += 1;
            }
        }

        //std::cout << "in an event\n";
        
    }
    return trueElems;
}

std::vector<std::string>& eventExtractor( pt::ptree &eventFile ) {

    std::vector<std::string> procElems;
    static std::vector<std::string> trueElems;

    // ZW: looping over children nodes of LHE file, but need to
    // keep track of event ordering, so we create a dummy loop
    // variable to remember current event number
    for (auto event : eventFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        // ZW: for each new event, find the linebreak from the first line (event information)
        // where it switches to the second line (first real particle line)
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

std::vector<std::vector<double>*>& singleEventParser( pt::ptree& eventFile, std::vector<bool>& relEv, unsigned int nEvt, unsigned int nPrt ) {
    bool getGs = true;
    unsigned int nuEvt = nEvt + ((32 - ( nEvt % 32 )) % 32);
    static std::vector<double> momVector( 4 * nuEvt * nPrt);
    static std::vector<double> alphaVector( nuEvt );
    std::vector<std::string> procElems;
    unsigned int momIndex = 0;
    unsigned int alphaIndex = 0;
    unsigned int currEvt = 0;

    for (auto event : eventFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        if (relEv[currEvt] ) {
        std::replace( event.second.data().begin(), event.second.data().end(), '\n', ' ');
        boost::split(procElems, event.second.data(), boost::is_any_of(" "));
        procElems.erase(std::remove(procElems.begin(), procElems.end(), ""));
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
        if( getGs ){
            alphaVector[alphaIndex] = std::sqrt( 4.0 * M_PI * std::stod(procElems[5]));
        } else {
            alphaVector[alphaIndex] = std::stod(procElems[5]);
        }
        alphaIndex += 1;
        }
        currEvt += 1;
    }
    static std::vector<std::vector<double>*> ptrVec{ &momVector, &alphaVector };
    return ptrVec;
}

std::vector<std::string>& stringSplitter( std::string& currEvent ){
    static std::vector<std::string> procElems;
    std::replace( currEvent.begin(), currEvent.end(), '\n', ' ');
    boost::split(procElems, currEvent, boost::is_any_of(" "));
    procElems.erase(std::remove(procElems.begin(), procElems.end(), ""));
    return procElems;
}

std::string& procReader( std::string& currEvent ){
    std::vector<std::string> eventElems = stringSplitter( currEvent );
    static std::string process = eventElems[0];
    process += ": ";
    unsigned int nPrt = std::stoi(eventElems[0]);
    bool prtStatus = true;
    for( unsigned int prtcl = 0; prtcl < nPrt; ++prtcl)
    {
        process += eventElems[6 + 13*prtcl] + " ";
        if ( prtStatus ){
            if ( eventElems[7 + 13*prtcl] != eventElems[7 + 13*(prtcl+1)]){
                process += "> ";
                prtStatus = false;
            }
        }
    } 
    return process;
}

std::vector<std::vector<bool>*>& procOrder( pt::ptree& eventFile, std::vector<std::string> evtSet, unsigned int nEvt ) {
    static std::vector<std::vector<bool>*> eventBools( evtSet.size() );
    for ( auto vecPtr : eventBools )
    {
        vecPtr->resize( nEvt );
    }
    unsigned int currEv = 0;

    for (auto event : eventFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        std::string currProc = procReader( event.second.data() );
        for ( unsigned int k = 0; k < evtSet.size(); ++k) {
            if ( currProc == evtSet[k] )
            {
                (*eventBools[k])[currEv] = true;
            } else {
                (*eventBools[k])[currEv] = false;
            }
        }
        currEv += 1;
    }
    return eventBools;
}

std::vector<std::string>& processExtractor( pt::ptree& eventFile ) {
    static std::vector<std::string> processes;
    for (auto event : eventFile.get_child("LesHouchesEvents")) {
        if (event.first != "event"){
            continue;
        }
        std::string currProc = procReader( event.second.data() );
        if ( std::find(processes.begin(), processes.end(), currProc) == processes.end()) {
            processes.push_back(currProc);
        }
    }
    return processes;
}

std::vector<std::vector<double>*>& multiEventParser( pt::ptree& eventFile ){
    std::cout << "\n\nin multiEv\n\n";
    std::vector<std::string> procList = processExtractor( eventFile );
    std::cout << "\n\nextracted processes\n\n";
    std::vector<unsigned int> numPrts(procList.size());
    std::cout <<
    for ( unsigned int k = 0; k < procList.size(); ++k )
    {
        //std::cout << "\n\n" << procList[k] << "\n\n";
        numPrts[k] = std::stoi(procList[k].substr(0,1));
    }
    std::cout << "\n\ngot prtnos\n\n";
    static std::vector<std::vector<double>*> vecPtrs;
    unsigned int nEvt = noEvt( eventFile );
    std::cout << "\n\nfound nEvt\n\n";
    std::vector<std::vector<bool>*> procOrdering = procOrder( eventFile, procList, nEvt );
    std::cout << "\n\nfound event ordering\n\n";
    for (unsigned int k = 0; k < procList.size(); ++k )
    {
        std::cout << "\n\n" << numPrts[k] << "\n\n" << nEvt << "\n\n";
        auto processVecs = singleEventParser( eventFile, *procOrdering[k], nEvt, numPrts[k] );
        vecPtrs.insert(std::end(vecPtrs), std::begin(processVecs), std::end(processVecs) );
    }
    return vecPtrs;
}

std::vector<std::vector<double>*>& lheParser( std::string fileName ){
    pt::ptree lheFile = fileLoader( fileName );
    return multiEventParser(lheFile);
}

}
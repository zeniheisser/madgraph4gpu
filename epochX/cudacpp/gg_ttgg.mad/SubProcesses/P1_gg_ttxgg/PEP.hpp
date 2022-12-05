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

std::vector<std::string>& procList ( pt::ptree eventFile ) {

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

std::set<std::pair<std::string, int>>& procExtractor ( pt::ptree eventFile ) {

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

std::vector<std::string>& pepSplitter ( pt::ptree eventFile ) {

    // ZW: get the generator
    // development used MG5aMC so could make simplifications for that,
    // but the extraction works for general LHE files 
    auto genName = eventFile.get<std::string>("LesHouchesEvents.init.generator.<xmlattr>.name");

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
        auto startPos = event.second.data().find("\n", 8); 
        auto noPrts = std::stoi(event.second.data().substr(0,7));
        std::replace( event.second.data().begin(), event.second.data().end(), '\n', ' ');
        boost::split(procElems, event.second.data(), boost::is_any_of(" "));
        int falseSize = std::count(procElems.begin(), procElems.end(), "");
        trueElems.reserve(procElems.size() - falseSize);
        for( int k = 0; k < procElems.size(); ++k){
            if( procElems[k] != ""){
                trueElems.push_back(procElems[k]);
            }
        }

        //std::cout << "in an event\n";
        
    }
    return trueElems;
}

}
#include "Shell.h"

#include "UAL/Common/Def.hh"
#include "UAL/APDF/APDF_Builder.hh"

#include "UAL/SMF/AcceleratorNodeFinder.hh"
#include "SMF/PacLattElement.h"
#include "SMF/PacElemAttributes.h"
#include "SMF/PacElemMultipole.h"
#include "PAC/Beam/Bunch.hh"

#include "Main/Teapot.h"
#include "Main/TeapotTwissService.h"
#include "TEAPOT/Integrator/MapDaIntegrator.hh"


COSY::UAL1::Shell::Shell() {
}

bool COSY::UAL1::Shell::use(const UAL::Arguments& arguments) {

    bool result = UAL::Shell::use(arguments);
    if (!result) return result;

    PacLattices::iterator it = PacLattices::instance()->find(m_accName);
    if (it == PacLattices::instance()->end()) {
        std::cerr << "There is no " << m_accName << " accelerator " << endl;
        return false;
    }

    p_lattice = &(*it);

    return true;
}


void COSY::UAL1::Shell::addN(const std::string& name, double value)
{

    if (p_lattice == 0) {
        std::cerr << "There is no " + m_accName << " accelerator " << endl;
        return;
    }

    PacLattice& lattice = *p_lattice;

    for (int i = 0; i < lattice.size(); i++) {
      if(lattice[i].getDesignName() == name){
	lattice[i].addN(value);
      }
    }
}

void COSY::UAL1::Shell::addMadK1K2(const std::string& name, double k1, double k2)
{

    if (p_lattice == 0) {
        std::cerr << "There is no " + m_accName << " accelerator " << endl;
        return;
    }

    PacLattice& lattice = *p_lattice;

    for (int i = 0; i < lattice.size(); i++) {
      if(lattice[i].getDesignName() == name){

	// std::cout << name << std::endl;

	PacElemAttributes* body = lattice[i].getBody();
	if(body == 0) continue;

	double l = lattice[i].getLength();

	PacElemMultipole mlt(2);
        mlt.kl(1) = k1*l;
        mlt.kl(2) = k2/2.*l;
	body->add(mlt);
      }
    }

}

void COSY::UAL1::Shell::calculateTwiss() {
    if (p_lattice == 0) {
        std::cerr << "There is no " + m_accName << " accelerator " << endl;
        return;
    }

    PAC::BeamAttributes& ba = getBeamAttributes();

    PacVTps oneTurnMap;
    std::vector<PacVTps> maps;

    calculateMaps(oneTurnMap, maps, 1, ba);

    calculateTwiss(oneTurnMap, maps, m_twissVector);

}

void COSY::UAL1::Shell::writeTwissToFile(const char* fileName) {

    if (p_lattice == 0) {
        std::cerr << "There is no " + m_accName << " accelerator " << endl;
        return;
    }

    PacLattice& lattice = *p_lattice;

    std::ofstream out(fileName);

    std::vector<std::string> columns(11);
    columns[0] = "#";
    columns[1] = "name";
    columns[2] = "suml";
    columns[3] = "betax";
    columns[4] = "alfax";
    columns[5] = "mux";
    columns[6] = "dx";
    columns[7] = "betay";
    columns[8] = "alfay";
    columns[9] = "muy";
    columns[10] = "dy";

    char line[200];
    char endLine = '\0';

    double twopi = 2.0 * UAL::pi;

    sprintf(line, "Number of elements: %5d", m_twissVector.size());
    out << line << std::endl;

    out << "------------------------------------------------------------";
    out << "------------------------------------------------------------" << std::endl;

    sprintf(line, "%-5s %-10s   %-15s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s%c",
            columns[0].c_str(), columns[1].c_str(), columns[2].c_str(), columns[3].c_str(),
            columns[4].c_str(),
            columns[5].c_str(), columns[6].c_str(), columns[7].c_str(), columns[8].c_str(),
            columns[9].c_str(), columns[10].c_str(), endLine);
    out << line << std::endl;

    out << "------------------------------------------------------------";
    out << "------------------------------------------------------------" << std::endl;

    int i = 0;
    std::string initName = "initial";
    double position = 0.0;

    sprintf(line, "%5d %-10s %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e%c",
            i, initName.c_str(), position,
            m_twissVector[i].beta(0), m_twissVector[i].alpha(0),
            m_twissVector[i].mu(0) / twopi, m_twissVector[i].d(0),
            m_twissVector[i].beta(1), m_twissVector[i].alpha(1),
            m_twissVector[i].mu(1) / twopi, m_twissVector[i].d(1), endLine);
    out << line << std::endl;

    for (int i = 1; i < lattice.size(); i++) {

        PacLattElement& el = lattice[i - 1];

        position = position + el.getLength();

        sprintf(line, "%5d %-10s %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e%c",
                i, el.getDesignName().c_str(), position,
                m_twissVector[i].beta(0), m_twissVector[i].alpha(0),
                m_twissVector[i].mu(0) / twopi, m_twissVector[i].d(0),
                m_twissVector[i].beta(1), m_twissVector[i].alpha(1),
                m_twissVector[i].mu(1) / twopi, m_twissVector[i].d(1), endLine);
        out << line << std::endl;
    }

    out.close();
}

void COSY::UAL1::Shell::calculateMaps(PacVTps& oneTurnMap,
        std::vector<PacVTps>& maps,
        int order,
        PAC::BeamAttributes& ba) {
    if (p_lattice == 0) {
        std::cerr << "There is no " + m_accName << " accelerator " << endl;
        return;
    }

    PacLattice& lattice = *p_lattice;

    maps.resize(lattice.size());

    PAC::Position orbit;
    // m_teapot->clorbit(orbit, m_ba);

    PacTMap map(6);
    map.refOrbit(orbit);

    map.setCharge(ba.getCharge());
    map.setMass(ba.getMass());
    map.setEnergy(ba.getEnergy());

    int mltOrder = map.mltOrder();
    map.mltOrder(order);

    UAL::PropagatorSequence& seq = m_ap->getRootNode();
    UAL::PropagatorIterator ip;

    int it = 0;

    for (ip = seq.begin(); ip != seq.end(); ip++) {

      // std::cout << it << " " << (*ip)->getFrontAcceleratorNode().getName() << " " << (*ip)->getFrontAcceleratorNode().getType() << "\n";

        PacTMap smap(6);

        smap.refOrbit(orbit);
        smap.setCharge(ba.getCharge());
        smap.setMass(ba.getMass());
        smap.setEnergy(ba.getEnergy());

        // (*ip)->propagate(map);

        (*ip)->propagate(smap);
        maps[it] = static_cast<PacVTps&> (smap);

        map *= smap;

        it++;

    }

    oneTurnMap = static_cast<PacVTps&> (map);

    map.mltOrder(mltOrder);

}

void COSY::UAL1::Shell::calculateTwiss(const PacVTps& oneTurnMap,
        const std::vector<PacVTps>& maps,
        std::vector<PacTwissData>& twissVector) {

    Teapot* teapot = UAL::OpticsCalculator::getInstance().m_teapot;

    PAC::Position orbit;
    // m_teapot->clorbit(orbit, m_ba);

    PacTwissData twiss;

    TeapotTwissService tservice(*teapot);
    tservice.define(twiss, oneTurnMap);

    twissVector.resize(maps.size());

    double mux = 0.0, muy = 0.0;
    twiss.mu(0, mux);
    twiss.mu(1, muy);

    for (unsigned int it = 0; it < maps.size(); it++) {

        twissVector[it] = twiss;

        teapot->trackTwiss(twiss, maps[it]);

        if ((twiss.mu(0) - mux) < 0.0) twiss.mu(0, twiss.mu(0) + 1.0);
        mux = twiss.mu(0);

        if ((twiss.mu(1) - muy) < 0.0) twiss.mu(1, twiss.mu(1) + 1.0);
        muy = twiss.mu(1);

        // twissVector[it] = twiss;
    }

}

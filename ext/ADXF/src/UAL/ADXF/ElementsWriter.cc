#include "UAL/ADXF/ElementsWriter.hh"
#include "UAL/ADXF/elements/MarkerWriter.hh"
#include "UAL/ADXF/elements/DriftWriter.hh"
#include "UAL/ADXF/elements/SbendWriter.hh"
#include "UAL/ADXF/elements/QuadrupoleWriter.hh"
#include "UAL/ADXF/elements/SextupoleWriter.hh"
#include "UAL/ADXF/elements/MultipoleWriter.hh"
#include "UAL/ADXF/elements/HkickerWriter.hh"
#include "UAL/ADXF/elements/VkickerWriter.hh"
#include "UAL/ADXF/elements/KickerWriter.hh"
#include "UAL/ADXF/elements/HmonitorWriter.hh"
#include "UAL/ADXF/elements/VmonitorWriter.hh"
#include "UAL/ADXF/elements/MonitorWriter.hh"
#include "UAL/ADXF/elements/InstrumentWriter.hh"
#include "UAL/ADXF/elements/RfCavityWriter.hh"
#include "UAL/ADXF/elements/RcollimatorWriter.hh"
#include "UAL/ADXF/elements/SolenoidWriter.hh"

UAL::ADXFElementsWriter::ADXFElementsWriter()
{
  m_elemWriters["Marker"]      = new UAL::ADXFMarkerWriter();
  m_elemWriters["Drift"]       = new UAL::ADXFDriftWriter();
  m_elemWriters["Sbend"]       = new UAL::ADXFSbendWriter();
  m_elemWriters["Quadrupole"]  = new UAL::ADXFQuadrupoleWriter();
  m_elemWriters["Sextupole"]   = new UAL::ADXFSextupoleWriter();
  m_elemWriters["Multipole"]   = new UAL::ADXFMultipoleWriter();
  m_elemWriters["Hkicker"]     = new UAL::ADXFHkickerWriter();
  m_elemWriters["Vkicker"]     = new UAL::ADXFVkickerWriter();
  m_elemWriters["Kicker"]      = new UAL::ADXFKickerWriter();
  m_elemWriters["Hmonitor"]    = new UAL::ADXFHmonitorWriter();
  m_elemWriters["Vmonitor"]    = new UAL::ADXFVmonitorWriter();
  m_elemWriters["Monitor"]     = new UAL::ADXFMonitorWriter();
  m_elemWriters["Instrument"]  = new UAL::ADXFInstrumentWriter();
  m_elemWriters["Rcollimator"] = new UAL::ADXFRcollimatorWriter();
  m_elemWriters["Solenoid"]    = new UAL::ADXFSolenoidWriter();
  m_elemWriters["RfCavity"]    = new UAL::ADXFRfCavityWriter();
}

UAL::ADXFElementsWriter::~ADXFElementsWriter()
{
  std::map<std::string, ADXFElementWriter*>::iterator it; 
  for(it = m_elemWriters.begin(); it != m_elemWriters.end(); it++){
    delete it->second;
    it->second = 0;
  }
}

// Write an element into an output stream.
void UAL::ADXFElementsWriter::writeDesignElements(ostream& out, const string& tab) 
{
  out << tab << "<elements>" << endl;

  std::string elemTab = tab + tab;

  PacGenElements* genElements =  PacGenElements::instance();

  PacGenElements::iterator it;
  std::map<std::string, UAL::ADXFElementWriter*>::iterator iw;

  for(it = genElements->begin(); it != genElements->end(); it++){
    iw = m_elemWriters.find(it->type());
    if(iw !=  m_elemWriters.end()){
      iw->second->writeDesignElement(out, *it, elemTab);
    } else {
      out << elemTab << it->name() << " " << it->type() << std::endl;
    }
  }


  out << tab << "</elements>" << endl;
}

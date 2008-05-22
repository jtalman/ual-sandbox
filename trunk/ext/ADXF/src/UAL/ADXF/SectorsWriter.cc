#include "UAL/ADXF/SectorsWriter.hh"

UAL::ADXFSectorsWriter::ADXFSectorsWriter()
{
}

UAL::ADXFSectorsWriter::~ADXFSectorsWriter()
{
}

// Write an element into an output stream.
void UAL::ADXFSectorsWriter::writeSectors(ostream& out, const string& tab) 
{
  out << tab << "<sectors>" << endl;

  std::string sectorTab = tab + tab;

  PacLattices* lattices =  PacLattices::instance();

  PacLattices::iterator it;
  for(it = lattices->begin(); it != lattices->end(); it++){
    m_sectorWriter.writeSector(out, *it, sectorTab);
  }

  out << tab << "</sectors>" << endl;
}

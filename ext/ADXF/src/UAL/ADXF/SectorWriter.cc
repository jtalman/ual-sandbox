#include "UAL/ADXF/SectorWriter.hh"

UAL::ADXFSectorWriter::ADXFSectorWriter()
{
}


UAL::ADXFSectorWriter::~ADXFSectorWriter()
{
}


void UAL::ADXFSectorWriter::writeSector(ostream& out, 
					PacLattice& lattice,
					const std::string& tab)
{
  out << tab << "<sector" 
      << " name=" << '\"' << lattice.name() << '\"';      
  out << " >" << std::endl; 

  double at = 0.0;
  std::string frameTab = tab + "  ";

  for(int i=0; i < lattice.size(); i++){
    if(lattice[i].getType().size() != 0 ) {
      out << frameTab << "<frame" 
	  << " ref=" << '\"' << lattice[i].getDesignName() << '\"'
	  << " at=" << '\"' << at << '\"';
      out << " />" << std::endl; 
    }
    at += lattice[i].getLength();
  }
  out << tab << "</sector>" << std::endl;

}




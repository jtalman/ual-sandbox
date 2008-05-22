#include <ctype.h>

#include "ual_sxf/Element.hh"
#include "ual_sxf/AcceleratorReader.hh"
#include "ual_sxf/NodeRegistry.hh"

string UAL_SXF_AcceleratorReader::empty_string;

// Constructor.
UAL_SXF_AcceleratorReader::UAL_SXF_AcceleratorReader(SXF::OStream& out, 
						     PacSmf& smf)
  : SXF::AcceleratorReader(out), m_refSMF(smf)
{
  m_pNodeRegistry = UAL_SXF_NodeRegistry::getInstance(out, smf);
}

// Write data.
void UAL_SXF_AcceleratorReader::write(ostream& out)
{
  UAL_SXF_Element *element = 0;
  UAL_SXF_NodeRegistry* registry = (UAL_SXF_NodeRegistry*) m_pNodeRegistry;

  char prefix = '_'; 
  string tab = "  ";
  double position, at, length;
  double delta = 0.0;
  int key;

  PacLattices* lattices = m_refSMF.lattices();

  // Loop lattices.
  int counter = 0;

  PacLattices::iterator it;
  for(it = lattices->begin(); it != lattices->end(); it++){

    PacLattice& lattice = (*it);
    out << lattice.name() << " sequence { " << endl;
 
    position = 0.0;
    for(int i = 0; i < (*it).size(); i++){

      // if( !lattice[i].name().compare(prefix, 0, 1) 
      // || !lattice[i].genElement().name().compare(prefix, 0, 1))

      UAL::AcceleratorNode* node = &(lattice[i]);

      if(lattice[i].name()[0] == prefix || 
        lattice[i].genElement().name()[0] == prefix)
      { 
	// Skip an internal drift 	
	delta = lattice[i].getLength();
        position += delta;
      }
      else{

	// Find 'at' position
	length = lattice[i].getLength();

	if(delta) { at = position + length/2.;}
	else { at = 0.0; }

	// Find the corresponding element adaptor

	// Get the SMF element key
	key = lattice[i].key();  

	// Get the element type.
	string str = key_to_string(key);

	if(str.length()){ 
	  str.at(0) = (char) tolower((int) str.at(0));
	  str.at(2) = (char) tolower((int) str.at(2));
	}

	// Get an element adaptor
	element = (UAL_SXF_Element*) registry->getElement(str.c_str());

	// Write element data

	if(element){
	  element->write(out, lattice[i], at, tab); 
	}
	else {
	  cerr << "Error wrong type -- include error element" << endl;
	}
	
	// Increase position
	delta = 0.0;
	position += length;
      }
    }
    out << "\nendsequence ";
    if(position) out << "at = " << position;
    out << "\n}" << endl;
  }
}

// Map the SMF element keys to their indecies in the collection of 
// element adaptors.
const string& UAL_SXF_AcceleratorReader::key_to_string(int key) const
{
  PacElemKeys::iterator it = m_refSMF.elemKeys()->find(key);
  if(it != m_refSMF.elemKeys()->end()){ return (*it).name();}
  return empty_string;
}

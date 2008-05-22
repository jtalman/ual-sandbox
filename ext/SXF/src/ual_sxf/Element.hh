//# Library     : SXF
//# File        : ual_sxf/Element.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEMENT_H
#define UAL_SXF_ELEMENT_H

#include "ual_sxf/Def.hh"


//
// The Element class is a container of the SXF bucket adaptors.
//

class UAL_SXF_Element : public SXF::Element
{
public:

  // Constructor.
  UAL_SXF_Element(SXF::OStream& out, const char* type, 
		  SXF::ElemBucket* bodyBucket, PacSmf& smf); 

  // Destructor.
  ~UAL_SXF_Element();

  // Open the element: Create the SMF lattice element. 
  virtual int openObject(const char* elementName, const char* elementType);

  // Updata the element: Add header attributes.
  virtual void update();

  // Clean up all temporary data.
  virtual void close();  

  // Add a bucket to the element.
  virtual void addBucket(SXF::ElemBucket* bucket);

  // Create the SMF generic element.
  virtual void setDesign(const char* name);

  // Set an element length
  void setLength(double l);

  // Set a longitudinal position of the node with 
  // respect to the beginning of the sequence.
  void setAt(double at);

  // Set a horizontal angle
  void setHAngle(double ha);

  // Set N
  void setN(double n);

  // Get an element length.
  double getLength() const;

  // Get a longitudinal position of the node with 
  // respect to the beginning of the sequence.
  double getAt() const;

  // Get a horizontal angle.
  double getHAngle() const;  

  // Get N.
  double getN() const;  

  // Return the lattice element.
  PacLattElement* getLattElement();

  // Write an element into an output stream.
  virtual void write(ostream& out, const PacLattElement& element, 
		     double at, const string& tab);

protected:

  // Header attributes.
  double m_dL, m_dAt, m_dHAngle, m_dN;

  // Pointer to a lattice element.
  PacLattElement* m_pElement;

  // Pointer to the SMF collection of generic elements.
  PacGenElements* m_pGenElements;

protected:

  // Write an element header.
  virtual void writeHeader(ostream& out, const PacLattElement& element, 
			   double at, const string& tab);

  // Write an element body.
  virtual void writeBody(ostream& out, const PacLattElement& element, 
			 const string& tab);

  // Write element common buckets
  virtual void writeCommonBuckets(ostream& out, 
				  const PacLattElement& element, 
				  const string& tab);

  // Map strings to the SMF element keys.
  int string_to_key(const char* str) const;  

};

#endif

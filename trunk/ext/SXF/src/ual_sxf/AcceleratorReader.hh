//# Library     : UalSXF
//# File        : ual_sxf/AcceleratorReader.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ACCELERATOR_H
#define UAL_SXF_ACCELERATOR_H

#include "ual_sxf/Def.hh"

//
// The AcceleratorReader class implements an interface for coordinating 
// SXF sequence and element adaptors. 
//  

class UAL_SXF_AcceleratorReader : public SXF::AcceleratorReader
{
public:

  // Constructor.
  UAL_SXF_AcceleratorReader(SXF::OStream& out, PacSmf& smf);

  // Write data.
  void write(ostream& out);

protected:

  // Reference to the SMF
  PacSmf& m_refSMF;

  // Empty string.
  static string empty_string;

protected:

  // Map the SMF element keys to their indecies 
  // in the collection of element adaptors.
  const string& key_to_string(int key) const;  

};

#endif

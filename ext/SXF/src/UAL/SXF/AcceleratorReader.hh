//# Library     : UAL
//# File        : UAL/SXF/AcceleratorReader.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ACCELERATOR_READER_HH
#define UAL_SXF_ACCELERATOR_READER_HH

#include "UAL/SXF/Def.hh"

namespace UAL {

  /** 
   * The AcceleratorReader class implements an interface for coordinating 
   * SXF sequence and element adaptors. 
   */ 

  class SXFAcceleratorReader : public SXF::AcceleratorReader
  {
  public:

    /** Constructor. */
    SXFAcceleratorReader(SXF::OStream& out, PacSmf& smf);

    /** Writes data. */
    void write(ostream& out);

  protected:

    /** Reference to the SMF */
    PacSmf& m_refSMF;

    /** Empty string.*/
    static string empty_string;

  protected:

    /** Map the SMF element keys to their indecies 
     * in the collection of element adaptors.
     */
  const string& key_to_string(int key) const;  

  };

}

#endif

//# Library     : UAL
//# File        : UAL/SXF/Sequence.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_SEQUENCE_HH
#define UAL_SXF_SEQUENCE_HH

#include "SMF/PacElemLength.h"

#include "UAL/SXF/Def.hh"


namespace UAL {

  /** 
   * The Sequence class implements a SXF adaptor to the SMF lattice.
   */

  class SXFSequence : public SXF::Sequence
  {
  public:

    /** Constructor.*/
    SXFSequence(SXF::OStream& out); 

    /** Destructor.*/
    ~SXFSequence();

    /** Create a sequence adaptor.*/
    SXF::Sequence* clone();

    /** Open sequence: Create a SMF lattice.
     * Return SXF_TRUE or SXF_FALSE.
     */
    int openObject(const char* name, const char* type);

    /** Update sequence: Copy a list of elements to a SMF lattice. */
    void update();

    /** Release data.*/
    void close();  

    /** Do nothing. */
    void setDesign(const char* name);

    /** Do nothing. */
    virtual void setLength(double l);

    /** Do nothing. */
    virtual void setAt(double l);

    /** Do nothing. */
    virtual void setHAngle(double h);

    /** Add an element (This version does not support nested sequences). */
    void addNode(SXF::AcceleratorNode* node);

  protected:

    /** Pointer to the SMF lattice. */
    PacLattice* m_pLattice;

    /** List of elements. */
    PacList<PacLattElement> m_ElementList;

    /** Drift counter. */
    int m_iDriftCounter;

    /** Current position.*/
    double m_dPosition;

    /** Drift bucket. */
    PacElemLength m_DriftLength; 

  };
}

#endif

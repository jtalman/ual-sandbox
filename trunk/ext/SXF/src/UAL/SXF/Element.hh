//# Library     : UAL
//# File        : UAL/SXF/Element.hh
//# Copyright   : see Copyrigh file
//# Author      : Nikolay Malitsky

#ifndef UAL_SXF_ELEMENT_HH
#define UAL_SXF_ELEMENT_HH

#include "UAL/SXF/Def.hh"

namespace UAL {

  /** 
   * The Element class is a container of the SXF bucket adaptors.
   */

  class SXFElement : public SXF::Element
  {
  public:

    /** Constructor.*/
    SXFElement(SXF::OStream& out, const char* type, 
	       SXF::ElemBucket* bodyBucket, PacSmf& smf); 

    /** Destructor. */
    ~SXFElement();

    /** Opens the element: Creates the SMF lattice element. */
    virtual int openObject(const char* elementName, const char* elementType);

    /** Updates the element: Adds header attributes. */
    virtual void update();

    /** Cleans up all temporary data.*/
    virtual void close();  

    /** Adds a bucket to the element.*/
    virtual void addBucket(SXF::ElemBucket* bucket);

    /** Create the SMF generic element. */
    virtual void setDesign(const char* name);

    /** Sets an element length */
	void setLength(double l);

    /** Sets a longitudinal position of the node with 
     * respect to the beginning of the sequence.
     */
    void setAt(double at);

    /** Sets a horizontal angle */
    void setHAngle(double ha);

    /** Sets N */
    void setN(double n);

    /** Returns an element length. */
    double getLength() const;

    /** Returns a longitudinal position of the node with 
     * respect to the beginning of the sequence.
     */
    double getAt() const;

    /** Returns  a horizontal angle. */
    double getHAngle() const;  

    /** Returns N.*/
    double getN() const;  

    /** Returns the lattice element. */
    PacLattElement* getLattElement();

    /** Writes an element into an output stream. */
    virtual void write(ostream& out, const PacLattElement& element, 
		       double at, const string& tab);

  protected:

    /** Header attributes.*/
    double m_dL, m_dAt, m_dHAngle, m_dN;

    /** Pointer to a lattice element. */
    PacLattElement* m_pElement;

    /** Pointer to the SMF collection of generic elements. */
    PacGenElements* m_pGenElements;

  protected:

    /** Writes an element header. */
    virtual void writeHeader(ostream& out, const PacLattElement& element, 
			     double at, const string& tab);

    /** Writes an element body. */
    virtual void writeBody(ostream& out, const PacLattElement& element, 
			   const string& tab);

    /** Writes element common buckets */
    virtual void writeCommonBuckets(ostream& out, 
				    const PacLattElement& element, 
				    const string& tab);

    /** Maps strings to the SMF element keys. */
    int string_to_key(const char* str) const;  

  private:

    int m_isNewDesignElement;

    void updateDesignElement();

  };

}

#endif

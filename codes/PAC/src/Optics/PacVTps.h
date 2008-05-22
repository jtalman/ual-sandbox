// Library     : PAC
// File        : Optics/PacVTps.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef PAC_VTPS_H
#define PAC_VTPS_H

#include "Templates/PacRCIPtr.h"
#include "Templates/PacMatrix.h"

#include "Optics/PacOpticsDef.h"
#include "ZLIB/Tps/VTps.hh"

#include "UAL/Common/Probe.hh"
#include "PAC/Beam/BeamAttributes.hh"
#include "PAC/Beam/Bunch.hh"

/**
Operates with Vectors of Truncated Power Series.
*/

class PacVTps : public PacRCIPtr< ZLIB::VTps >, public PAC::BeamAttributes, public UAL::Probe
{
public:

  // Constructors, destructor & copy operator

  /** Constructor. */
  PacVTps();

  /** Copy constructor. Copy contents of the ZLIB::VTps object to the contents of 
      the new PacVTps object and returns the pointer to this object. */
  PacVTps(const ZLIB::VTps& rhs);

  /** Copy constructor. Copy contents of the PacVTps object to the contents of 
      the new PacVTps object and returns the pointer to this object.*/
  PacVTps(const PacVTps& rhs);

  /** Copy operator. Copy contents of the ZLIB::VTps object to the contents of 
     the new PacVTps object and returns the pointer to this object.*/
  PacVTps& operator=(const ZLIB::VTps& rhs);

  /** Copy operator. Copy contents of the PacVTps object to the contents of 
      the new PacVTps object and returns the pointer to this object.*/
  PacVTps& operator=(const PacVTps& rhs); 

  /** Multiplies this and $rhs PacVTps objects and returns the pointer to 
      the new PacVTps object. */
  PacVTps& operator*=(const PacVTps& rhs); 

  // Access operators

  /** Returns the number of dimensions for Truncated Power Series.*/
  int size() const;

  // Order of truncated power series

  /** Returns the order of Truncated Power Series. */
  unsigned int order() const;

  /** Sets the order of Truncated Power Series. */
  void order(unsigned int o);  

  /** Returns the order for intermediate Tps components during multiplication procedure. */
  int mltOrder() const;

  /** Sets the order for intermediate Tps components during multiplication procedure. */  
  void mltOrder(int order);

  /** Defines ()-operator to access to the VTps components.*/
  double operator()  (int i, int j) const;

  /** Defines ()-operator to access to the VTps components.*/
  double& operator() (int i, int j);

  // I/0

  /** Reads the coefficients of VTps from a file with name "nfile". */
  void read(const char* nfile);
 
  /** Dumps the coefficients of VTps to a file with name "nfile". */
  void write(const char*  nfile);

protected:

  void create(const ZLIB::VTps& vtps) {
      PacRCIPtr<ZLIB::VTps>::operator=(new ZLIB::VTps(vtps)); 
  }

  void create(const PacVTps& vtps) {
      PacRCIPtr<ZLIB::VTps>::operator=(vtps); 
  }
};

#endif

// Program     : PAC
// File        : Optics/PacChromData.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef PAC_CHROM_DATA_H
#define PAC_CHROM_DATA_H

#include "Optics/PacTwissData.h"

/**
Operates with Chromaticity Data.
*/

class PacChromData
{

 public:

  // Constructor

/** Constructor. 
The variable dim defines the number of dimensions for which Chromaticity Data are stored.
*/
  PacChromData(int dim = 2);
  
/// Copy constructor.
  PacChromData(const PacChromData& right);

/// Destructor.
  virtual ~PacChromData();

/// Copy operator.
 PacChromData& operator = (const PacChromData& right);

 // Data Access

 enum Coordinates {
   W = 0,
   PHI,
   DMU,
   DD,
   DDP,
   SIZE
 };


/// Returns the number of dimensions.
 int dimension() const;

/**@name Methods deals with PacTwissData objects. */ 
//@{
///  Returns the reference to the PacTwissData object.
 PacTwissData&       twiss();
/// Returns the reference to the PacTwissData object.
 const PacTwissData& twiss() const;
/// Copies Chromaticity Parameters from the "v" PacTwissData objects to this object.
 void                twiss(const PacTwissData& v);
//@}

/**@name Methods  operates with the Chromaticity Data. */ 
//@{

/// Returns the w parameter for the particular dimension defined by the integer variable d.
 double& w(int d);  
/// Returns the w parameter for the particular dimension defined by the integer variable d.
 double  w(int d) const;
/// Sets the w parameter (the variable v) for the particular dimension defined by the integer variable d.
 void    w(int d, double v);

/// Returns the phi parameter for the particular dimension defined by the integer variable d.
 double& phi(int d);  
/// Returns the phi parameter for the particular dimension defined by the integer variable d.
 double  phi(int d) const;
/// Sets the phi parameter (the variable v) for the particular dimension defined by the integer variable d.
 void    phi(int d, double v);

/// Returns the dmu parameter for the particular dimension defined by the integer variable d.
 double& dmu(int d);  
/// Returns the dmu parameter for the particular dimension defined by the integer variable d.
 double  dmu(int d) const;
/// Sets the dmu parameter (the variable v) for the particular dimension defined by the integer variable d.
 void    dmu(int d, double v);

/// Returns the dd parameter for the particular dimension defined by the integer variable d.
 double& dd(int d);  
/// Returns the dd parameter for the particular dimension defined by the integer variable d.
 double  dd(int d) const;
/// Sets the dmu parameter (the variable v) for the particular dimension defined by the integer variable d.
 void    dd(int d, double v);

/// Returns the ddp parameter for the particular dimension defined by the integer variable d.
 double& ddp(int d);  
/// Returns the ddp parameter for the particular dimension defined by the integer variable d.
 double  ddp(int d) const;
/// Sets the ddp parameter (the variable v) for the particular dimension defined by the integer variable d.
 void    ddp(int d, double v);
//@}

 protected:

 PacTwissData twiss_;
 PacMatrix<double> data_;

 protected:

 double& value(int d, int index);
 double  value(int d, int index) const;

};

inline PacTwissData& PacChromData::twiss()               { return twiss_; }
inline const  PacTwissData& PacChromData::twiss() const  { return twiss_; }
inline void PacChromData::twiss(const PacTwissData& v)   { twiss_ = v; }

inline double& PacChromData::w(int d)            { return value(d, W); }
inline double PacChromData::w(int d) const       { return value(d, W); }
inline void PacChromData::w(int d, double v)     { value(d, W) = v; }

inline double& PacChromData::phi(int d)          { return value(d, PHI); }
inline double PacChromData::phi(int d) const     { return value(d, PHI); }
inline void PacChromData::phi(int d, double v)   { value(d, PHI) = v; }

inline double& PacChromData::dmu(int d)          { return value(d, DMU); }
inline double PacChromData::dmu(int d) const     { return value(d, DMU); }
inline void PacChromData::dmu(int d, double v)   { value(d, DMU) = v; }

inline double& PacChromData::dd(int d)           { return value(d, DD); }
inline double PacChromData::dd(int d) const      { return value(d, DD); }
inline void PacChromData::dd(int d, double v)    { value(d, DD) = v; }

inline double& PacChromData::ddp(int d)          { return value(d, DDP); }
inline double PacChromData::ddp(int d) const     { return value(d, DDP); }
inline void PacChromData::ddp(int d, double v)   { value(d, DDP) = v; }

#endif

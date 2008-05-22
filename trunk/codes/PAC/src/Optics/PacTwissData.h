// Program     : PAC
// File        : Optics/PacTwissData.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef PAC_TWISS_DATA_H
#define PAC_TWISS_DATA_H

#include "UAL/Common/Object.hh"
#include "Templates/PacMatrix.h"
#include "Optics/PacOpticsDef.h"

/**
Operates with Twiss Data.
*/


class PacTwissData : public UAL::Object
{
 public:

// Constructor
/// Constructor. The variable dim defines the number of dimensions for which Twiss Data are stored.

 PacTwissData(int dim = 2);

/// Copy constructor.
 PacTwissData(const PacTwissData& right);
 
/// Destructor.
 virtual ~PacTwissData();

/// Copy operator.
 PacTwissData& operator = (const PacTwissData& right);

 // Data Access

 enum Coordinates {
   BETA = 0,
   ALPHA,
   MU,
   D,
   DP,
   SIZE
 };

/// Returns the number of dimensions.
 int dimension() const;


/**@name Methods  operates with the Twiss Data. */ 
//@{
/// Returns the beta parameter for the particular dimension defined by the integer variable d.
 double& beta(int d);  
/// Returns the beta parameter for the particular dimension defined by the integer variable d. 
 double  beta(int d) const;
/// Sets the beta parameter (the variable v) for the particular dimension defined by the integer variable d.
 void    beta(int d, double v);

/// Returns the alpha parameter for the particular dimension defined by the integer variable d.
 double& alpha(int d);  
/// Returns the alpha parameter for the particular dimension defined by the integer variable d.
 double  alpha(int d) const;
/// Sets the alpha parameter (the variable v) for the particular dimension defined by the integer variable d.
 void    alpha(int d, double v);

/// Returns the mu parameter for the particular dimension defined by the integer variable d.
 double& mu(int d);  
/// Returns the mu parameter for the particular dimension defined by the integer variable d.
 double  mu(int d) const;
/// Sets the mu parameter (the variable v) for the particular dimension defined by the integer variable d.
 void    mu(int d, double v);

/// Returns the d parameter for the particular dimension defined by the integer variable d.
 double& d(int d);  
/// Returns the d parameter for the particular dimension defined by the integer variable d.
 double  d(int d) const;
/// Sets the d parameter (the variable v) for the particular dimension defined by the integer variable d.
 void    d(int d, double v);

/// Returns the dp parameter for the particular dimension defined by the integer variable d.
 double& dp(int d);  
/// Returns the dp parameter for the particular dimension defined by the integer variable d.
 double  dp(int d) const;
/// Sets the dp parameter (the variable v) for the particular dimension defined by the integer variable d.
 void    dp(int d, double v);

//@}

 protected:

 PacMatrix<double> data_;

 protected:

 double& value(int d, int index);
 double  value(int d, int index) const;

};

inline double& PacTwissData::beta(int d)        { return value(d, BETA); }
inline double PacTwissData::beta(int d) const   { return value(d, BETA); }
inline void PacTwissData::beta(int d, double v) { value(d, BETA) = v; }

inline double& PacTwissData::alpha(int d)        { return value(d, ALPHA); }
inline double PacTwissData::alpha(int d) const   { return value(d, ALPHA); }
inline void PacTwissData::alpha(int d, double v) { value(d, ALPHA) = v; }

inline double& PacTwissData::mu(int d)          { return value(d, MU); }
inline double PacTwissData::mu(int d) const     { return value(d, MU); }
inline void PacTwissData::mu(int d, double v)   { value(d, MU) = v; }

inline double& PacTwissData::d(int d)           { return value(d, D); }
inline double PacTwissData::d(int d) const      { return value(d, D); }
inline void PacTwissData::d(int d, double v)    { value(d, D) = v; }

inline double& PacTwissData::dp(int d)          { return value(d, DP); }
inline double PacTwissData::dp(int d) const     { return value(d, DP); }
inline void PacTwissData::dp(int d, double v)   { value(d, DP) = v; }

#endif

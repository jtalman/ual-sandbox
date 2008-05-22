// Library     : PAC
// File        : Survey/PacSurvey.h
// Description : The class PacSurvey represents the vector of coordinates
//               and angles that uniquely define the location and
//               directions of the local orbit in the global Cartesian
//               coordinate system
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef SURVEY_H
#define SURVEY_H

#include "Survey/PacSurveyDef.h"

/**
The class PacSurvey represents the vector of coordinates
and angles that uniquely define the location and
directions of the local orbit in the global Cartesian
coordinate system.
*/


class PacSurvey
{
public:

  // Constructors & destructor

/// Constuctor.
  PacSurvey()                               {create(); initialize();}
/// Copy constructor.
  PacSurvey(const PacSurvey& s)             {create(); initialize(s);}
/// Destructor.
 ~PacSurvey()                               {if(data) delete [] data;}

  // Assingment operators
/// Assingment operator.
  void operator=(const PacSurvey& s)        {initialize(s);}

  // Interface. Data can be accessed via direct methods or be represented as a hash vector.
/**@name Interface.
   Data can be accessed via direct methods or be represented as a hash vector.
*/
//@{

  // Global Cartesian coordinates
  
 /**@name Global Cartesian coordinates. */
 //@{

  /// returns x-coordinate [m]
  double& x()            { return data[0]; } 
  /// returns x-coordinate  [m]
  double  x() const      { return data[0]; }
  /// sets x-coordinate [m]
  void    x(double v)    { data[0] = v;    }
  /// returns y-coordinate [m]
  double& y()            { return data[1]; }
  /// returns y-coordinate [m]
  double  y() const      { return data[1]; }
  /// sets y-coordinate [m]
  void    y(double v)    { data[1] = v;    }
  /// returns z-coordinate [m]
  double& z()            { return data[2]; }
  /// returns z-coordinate [m]
  double  z() const      { return data[2]; }
  /// sets z-coordinate [m]
  void    z(double v)    { data[2] = v;    }
 
 //@}


  // The path length

 /**@name The path length. */
 //@{
  /// returns the total path length along the orbit [m].
  double& suml()         { return data[3]; }
  /// returns the total path length along the orbit [m].
  double  suml()  const  { return data[3]; }
  /// sets the total path length along the orbit [m].
  void    suml(double v) { data[3] = v;    }
 //@}
 
  // Angles of rotations (right hand rule):

 /**@name Angles of rotations (right hand rule). */
 //@{

  /// returns the angle around the global Y-axis
  double& theta()        { return data[4]; }   // around the global Y-axis 
  /// returns the angle around the global Y-axis
  double  theta() const  { return data[4]; }
  /// sets the angle around the global Y-axi
  void    theta(double v){ data[4] = v;    }
  /// returns the angle between the reference orbit and its projection on ZX plane 
  double& phi()          { return data[5]; }   // between the reference orbit and its projection on ZX plane
  /// returns the angle between the reference orbit and its projection on ZX plane 
  double  phi() const    { return data[5]; }
  /// sets the angle between the reference orbit and its projection on ZX plane 
  void    phi(double v)  { data[5] = v;    }
  /// returns the angle around the local s-axis  
  double& psi()          { return data[6]; }   // around the local s-axis
  /// returns the angle around the local s-axis  
  double  psi() const    { return data[6]; }
  /// sets the angle around the local s-axis  
  void    psi(double v)  { data[6] = v;    }
 //@}


  // Vector or Hash vector (x, y, z, suml, theta, phi, psi)

 /**@name Vector (0..6) or Hash vector (x, y, z, suml, theta, phi, psi). */
 //@{
  ///
  double& operator[](int i)                  {check(i); return data[i];}
  ///
  double  operator[](int i) const            {check(i); return data[i];}

  ///
  double& operator[](const char* id)         {return data[index(id)];}      
  ///
  double  operator[](const char* id) const   {return data[index(id)];}
 //@}

//@}
// End of Interface.

  /// returns the number of variables (default 7)
  int  size() const                          { return _size; }
  /// sets all variables equal to 0.0
  void erase()                               { initialize(); }
 

protected:

  int       _size;
  double*    data;

private:

  void create();
  void initialize();
  void initialize(const PacSurvey& s);
  void check(int index) const;
  int  index(const char* id) const;

};

#endif

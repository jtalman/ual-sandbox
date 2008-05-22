// Program     : PAC
// File        : PAC/Beam/Spin.hh
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef UAL_PAC_SPIN_HH
#define UAL_PAC_SPIN_HH

#include "UAL/Common/AttributeSet.hh"

namespace PAC {

  /** Container of spin coordinates.
  */

  class Spin  : public UAL::AttributeSet 
    {
    public:

      // Constructors &  destructor

      /** Constructor; creates the vertical spin (0, 1, 0) */
      Spin();

      /** Constructor; creates the spin (sx, sy, sz) */
      Spin(double sx, double sy, double sz);

      /** Copy constructor */
      Spin(const Spin& s);

      /** Destructor */
      virtual ~Spin();

      /** Copy operator */
      const Spin& operator =(const Spin& s);

      // Access methods

      /** Returns the x-coordinate. */ 
      double  getSX() const;

      /** Sets the x-coordinate. */
      void    setSX(double v);
  
      /** Returns the y-coordinate. */
      double  getSY() const;

      /** Sets the y-coordinate. */
      void    setSY(double v);

      /** Returns the z-coordinate. */
      double  getSZ() const;

      /** Sets the z-coordinate [m]. */
      void    setSZ(double v);

      /** Sets spin coordinates.*/
      void set(double sx, double sy, double sz);

      /** Returns the coordinate specified by index */
      double  operator[](int index) const;          

      /** Returns the number of coordinates */  
      int size() const;

    protected:
    
      /** Spin coordinates */
      double m_sx, m_sy, m_sz;
 
    };
}

#endif
        

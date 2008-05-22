// Library     : ZLIB
// File        : ZLIB/Tps/Space.hh
// Description : Base class for Tps and VTps. 
//               It handles the unique instance of GlobalTable.            
// Copyright   : see Copyright file
// Author      : Yiton Yan
// C++ version : Nikolay Malitsky

#ifndef UAL_ZLIB_SPACE_HH
#define UAL_ZLIB_SPACE_HH

#include "ZLIB/Tps/GlobalTable.h"

namespace ZLIB {

/** Base class for Tps and VTps objects. It defines the space of Truncated Power Series 
    and contains a global container of One-Step Index Pointers.
 */

  class Space
    {

    public:

      /** Constructor. Defines the space of Tps objects and initializes the global container 
	  of One-Step Index Pointers. The constructor has two arguments: the phase-space 
	  dimension (dim) and maximum order (maxOrder) of Tps objects.
      */
      Space(unsigned int dim, unsigned int maxOrder);

      /** Destructor */
      virtual ~Space();

      // Attributes

      /** Returns the phase-space dimension of Tps objects */
      unsigned int dimension() const;

      /** Returns the maximum order of Tps objects. */
      unsigned int maxOrder() const;

      /** Returns the order of the Tps multiplication product. */
      unsigned int mltOrder() const;

      /** Sets the order of the Tps multiplication product. */
      void mltOrder(unsigned int o);

      /** Returns a number of monomials for the specifed order */
      unsigned int nmo(unsigned int order) const;

    protected:

      /** global container of One-Step Index Pointers */
      static GlobalTable* _table;

      /** order of the Tps multiplication product*/
      static unsigned int _mltOrder;

    protected:

      /** Constructor */
      Space();

      /** Returns min of a and b values */
      int min(int a, int b) const { return a<b ? a : b; }

      /** Returns max of a and b values */
      int max(int a, int b) const { return a>b ? a : b; }  

    private:

      Space& operator=(const Space& ) {return *this;}
  
    };
}

inline unsigned int ZLIB::Space::dimension() const { return _table->dimension; }
inline unsigned int ZLIB::Space::maxOrder() const { return _table->order; }
inline unsigned int ZLIB::Space::mltOrder() const { return _mltOrder; }
inline unsigned int ZLIB::Space::nmo(unsigned int order) const { return _table->nmo[order]; }



#endif


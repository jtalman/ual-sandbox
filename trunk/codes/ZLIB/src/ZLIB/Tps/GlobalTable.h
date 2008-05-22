// Library     : ZLIB
// File        : ZLIB/Tps/GlobalTable.h
// Description : GlobalTable includes predefined indexes for efficient
//               TPS computation. All data structures and algorithms are 
//               based on Zlib Fortran subroutines.
// Copyright   : see Copyright file
// Author      : Yiton Yan
// C++ version : Nikolay Malitsky



#ifndef UAL_ZLIB_GLOBAL_TABLE_H
#define UAL_ZLIB_GLOBAL_TABLE_H

#include "ZLIB/Tps/Def.hh"

namespace ZLIB {

  class Space;

  /** Container of predefined indexes for efficient TPS computation. 
      All data structures and algorithms are based on Zlib Fortran subroutines.
  */

  struct GlobalTable
  {

    public:

    // Dimension and order
    int dimension;
    int order;

    // Tps data structures
  
    int*  nmo; // nmo[Omega] - number of monomials for order Omega 
    int** jv;

    // Multiplication

    int*    kp;
    int*    lp;
    int**   ikp;
    int**   ikb;

    // Derivative

    int**   jd; 

    // nmov --> jv --> js --> jpek() --> Static Members for Mult.
    //          jv --> js --> jpec() --> Static Members for Deriv.

    int***  nmov;
    int*    js1;
    int     nkpm;

    // VTps data structures

    // Multiplication
 
    int*    jpc;
    int*    ivpc;
    int*    ivppc;

    // Tracking

    int*    ivp;
    int*    jpp;

    // mp --> js2, ztpapek()--> jpek() --> pntcct() --> Static Members for Mult.
    // nmvo --> Static Members for Tracking

    int***  mp;
    int*    js2;
    int**   nmvo;

    void pntcct(int no);
   
    private:

    friend class Space;

    // Constructor & destructor

    GlobalTable() {}
    GlobalTable(int dim, int order);
    ~GlobalTable();

    GlobalTable& operator=(const GlobalTable& ) { return *this;}

    // Counter

    int counter;

    // Tps "subroutines"

    void defineTpsData(int dim, int order);
    int jpek(int* jps);
    void eraseTpsData();

    // VTps "subroutines"

    void defineVTpsData(int dim, int order);
    void ztpapek(int jtpa, int no);
    void eraseVTpsData();

  };
}

#endif

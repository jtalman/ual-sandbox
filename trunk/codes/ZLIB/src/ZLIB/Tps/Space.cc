// Library     : ZLIB
// File        : ZLIB/Tps/Space.cc
// Copyright   : see Copyright file
// Author      : Yiton Yan
// C++ version : Nikolay Malitsky

#include "ZLIB/Tps/Space.hh"

ZLIB::GlobalTable* ZLIB::Space::_table = 0;
unsigned int ZLIB::Space::_mltOrder = 0;

ZLIB::Space::Space()
{
  if(!_table){
    cerr << "Error : ZLIB::Space::Space() : ZLIB::Space  is not defined \n";
    assert(_table);
  }
  _table->counter++;
}


ZLIB::Space::Space(unsigned int dim, unsigned int order)
{
  if(!_table){
    _table = new ZLIB::GlobalTable(dim, order);
    _table->counter++;
    _mltOrder = order;
  }
  else{
    if(dimension() != dim){
      cerr << "Error : ZLIB::Space::define(unsigned int dim, unsigned int order) : "
	   << "attempt to redefine dimension \n";
      assert(dimension() == dim);
    }
    if(maxOrder() != order){
      cerr << "Error : ZLIB::Space::define(unsigned int dim, unsigned int order) : "
	   << "attempt to redefine max order \n";
      assert(maxOrder() == order);
    }   
  }      

}

ZLIB::Space::~Space()
{
  if(_table){
    if (--(_table->counter) <= 0) delete _table;
  }
}

void ZLIB::Space::mltOrder(unsigned int o)
{
  if(o > maxOrder()){
    cerr << "Error : ZLIB::Space::mltOrder(unsigned int o) : "
         << "o (" << o << ") > maxOrder(" << maxOrder() << ") \n";
    assert(0);
  }
  _mltOrder = o;
}

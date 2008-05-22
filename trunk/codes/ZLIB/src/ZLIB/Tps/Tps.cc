// Library     : ZLIB
// File        : ZLIB/Tps/Tps.cc
// Copyright   : see Copyright file
// Author      : Yiton Yan
// C++ version : Nikolay Malitsky & Alexander Reshetov

#include <cmath>
#include "ZLIB/Tps/Tps.hh"

// 5.2.00, Nikolay Malitsky
// operator+(const ZlibTps& rhs) : Replaced the code by return ZlibTps(rhs) += *this;
// operator*(const ZlibTps& p_zs) : Added ZlibTps zs(p_zs); 
// Temporary objects should be replaced by smart pointers.

// Constructors

ZLIB::Tps::Tps() : ZLIB::Space() { initialize(0.0, 0);  }
ZLIB::Tps::Tps(double c, unsigned int order) : ZLIB::Space() { initialize(c,order);  }
ZLIB::Tps::Tps(const ZLIB::Tps& rhs) : ZLIB::Space() { initialize(rhs); }

// Destructor

ZLIB::Tps::~Tps() { erase(); }

// Attributes 

void ZLIB::Tps::order(unsigned int o)
{
  if(_order == o ) return;

  checkOrder(o);

  double *tmp = new double[nmo(o)];
  if(!tmp){
    cerr << "Error: ZLIB::Tps::order(unsigned int o) : allocation failure \n";
    assert(tmp);
  } 
 
  unsigned int ms = min(nmo(o), size());
  unsigned int i;
  for(i=0;  i < ms; i++)     tmp[i] = _tps[i];
  for(i=ms; i < nmo(o); i++) tmp[i] =  0.0; 

  erase();

  _tps = tmp;
  _order = o;

} 


// Assignment operators

ZLIB::Tps& ZLIB::Tps::operator=(double c)
{
  if(_order != 0) {
    erase();
    initialize(c, 0);
  }
  else{
    _tps[0] = c;
  }
  return *this;
}

ZLIB::Tps& ZLIB::Tps::operator+=(double c)
{
  _tps[0] += c;
  return *this;
}

ZLIB::Tps& ZLIB::Tps::operator-=(double c) { return *this += -c; }

ZLIB::Tps& ZLIB::Tps::operator*=(double c) 
{
  unsigned int s = size();
  for(unsigned int i=0; i < s; i++) _tps[i] *= c;
  return *this;
}

ZLIB::Tps& ZLIB::Tps::operator/=(double c) { return *this *= 1./c; }

// Tps

ZLIB::Tps& ZLIB::Tps::operator=(const ZLIB::Tps& zs)
{
  unsigned int s = size(), zs_s = zs.size();

  if(s != zs_s){
    erase();
    initialize(zs);
  }
  else{
    for(unsigned int i=0; i < zs_s; i++) _tps[i] = zs._tps[i];
  }
  return *this;
}

ZLIB::Tps& ZLIB::Tps::operator+=(const ZLIB::Tps& zs)
{
  unsigned int s = size(), zs_s = zs.size();

  if(s < zs_s){

    double* tmp = new double[zs_s];
    if(!tmp){
      cerr << "Error : ZLIB::Tps::operator+=(const ZLIB::Tps& zs) : allocation failure \n";
      assert(tmp);
    }

    unsigned int i;
    for(i=0; i < zs_s; i++) tmp[i]  = zs._tps[i];
    for(i=0; i < s; i++)    tmp[i] += _tps[i];

    erase();

    _tps = tmp;
    _order = zs._order;    
  }
  else{
    for(unsigned int i=0; i < zs_s; i++) _tps[i] += zs._tps[i];
  }
  return *this;
}

ZLIB::Tps& ZLIB::Tps::operator-=(const ZLIB::Tps& rhs) { return *this += -rhs; }
ZLIB::Tps& ZLIB::Tps::operator*=(const ZLIB::Tps& rhs) { return *this  = (*this) * rhs; }
ZLIB::Tps& ZLIB::Tps::operator/=(const ZLIB::Tps& rhs) { return *this *= 1./rhs; }

// Additive & Multiplicative Operators

ZLIB::Tps ZLIB::Tps::operator+(const ZLIB::Tps& rhs) const
{ 
  // 5.2.00, NM
  // return _order > rhs._order ?  ZLIB::Tps(*this) += rhs : ZLIB::Tps(rhs) += *this; 
  return ZLIB::Tps(rhs) += *this; 
}

ZLIB::Tps ZLIB::Tps::operator-(const ZLIB::Tps& rhs) const { 
  return (*this) + (-rhs); 
}

ZLIB::Tps ZLIB::Tps::operator*(const ZLIB::Tps& p_zs) const
{
  ZLIB::Tps zs(p_zs); // 5.2.00, NM
  ZLIB::Tps tmp(0.0, min(_order + zs._order, mltOrder()));
  unsigned int t_s = tmp.size();

  if(!zs._order){
    for(unsigned int i=0; i < t_s; i++) tmp._tps[i] = _tps[i];
    return tmp *= zs._tps[0];
  }

  if(!_order){
    for(unsigned int i=0; i < t_s; i++) tmp._tps[i] = zs._tps[i]; 
    return tmp *= _tps[0];
  }

  // Order 0
  
  tmp._tps[0] = _tps[0]*zs._tps[0]; 

  if(!tmp._order) return tmp;

  // Order 1

  for(unsigned int i=1; i < nmo(1); i++)
    tmp._tps[i] = _tps[0]*zs._tps[i] + _tps[i]*zs._tps[0];


  if(tmp._order > 1){

    int noub;
    unsigned int noue;

    for(unsigned int j = nmo(1); j < nmo(tmp._order); j++) tmp._tps[j] = 0.;

    for(unsigned int io = 2; io <= tmp._order; io++){

      noub = io - zs._order;
      if(noub < 0) noub = 0;

      noue = _order;
      if(noue > io) noue = io;

      for(unsigned int j = nmo(io-1)+1; j <= nmo(io); j++)
	for(int i = _table->ikb[noub][j]; i <= _table->ikp[noue][j]; i++)
	  tmp._tps[j-1] += (_tps[_table->kp[i]]*zs._tps[_table->lp[i]]);	  
    }
  }
    
  return tmp;
}

ZLIB::Tps ZLIB::Tps::operator/(const ZLIB::Tps& rhs) const { return (*this) * (1./rhs); }

// Friends

ZLIB::Tps ZLIB::operator-(const ZLIB::Tps& rhs) { return ZLIB::Tps(rhs) *= -1.; }

ZLIB::Tps ZLIB::operator+(const ZLIB::Tps& rhs, double c) { return ZLIB::Tps(rhs) +=  c; }
ZLIB::Tps ZLIB::operator+(double c, const ZLIB::Tps& rhs) { return rhs + c; }
ZLIB::Tps ZLIB::operator-(const ZLIB::Tps& rhs, double c) { return ZLIB::Tps(rhs) += -c; }
ZLIB::Tps ZLIB::operator-(double c, const ZLIB::Tps& rhs) { return -(rhs - c); }
ZLIB::Tps ZLIB::operator*(const ZLIB::Tps& rhs, double c) { return ZLIB::Tps(rhs) *= c; }
ZLIB::Tps ZLIB::operator*(double c, const ZLIB::Tps& rhs) { return rhs * c; }
ZLIB::Tps ZLIB::operator/(const ZLIB::Tps& rhs, double c) { return ZLIB::Tps(rhs) *= (1./c); }

ZLIB::Tps ZLIB::operator/(double c, const ZLIB::Tps& zs)
{
  ZLIB::Tps el(zs);
  
  double el0 = el[0];
  if(fabs(el0) < ZLIB_TINY)
  {
    cerr << "Error: operator/(double c, const ZLIB::Tps& zs)  ";
    cerr << "fabs(zs[0])"  << " <  " << ZLIB_TINY << "\n";
    assert(fabs(el0) >= ZLIB_TINY);
  }

  double linearInv = 1/el0;

  ZLIB::Tps sum(linearInv, 0);

  el -= el0;
  el *= -1;

  unsigned int maxOrder = el.mltOrder();

  for(unsigned int i=1; i <= maxOrder; i++){
    el.mltOrder(i);
    sum *= el;
    sum += 1.;
    sum *= linearInv;
  }

  sum *= c;

  return sum;
  
}

ZLIB::Tps ZLIB::D(const ZLIB::Tps& tps, unsigned int iv) 
{
  if(iv >= tps.dimension()){
     cerr << "Error : D((const ZLIB::Tps& tps, unsigned int iv) : iv(";
     cerr << iv << ") >= " << tps.dimension() << "\n";
     assert(0);
  } 
  unsigned int   index = iv + 1; 

  ZLIB::Tps result(0.0, 0);

  if(!tps.order()) return result;
  
  result.order(tps.order() - 1);

  for(unsigned int i=1; i <= result.size(); i++)
    result._tps[i-1] = (tps._table->jv[index][i] + 1.)*tps._tps[ tps._table->jd[index][i] ];

  return result;  
}

ZLIB::Tps ZLIB::poisson(const ZLIB::Tps& tps1, const ZLIB::Tps& tps2) 
{
  unsigned int nd = tps1.dimension()/2;

  if(2*nd != tps1.dimension()){
    cerr << "Error : poisson(const ZLIB::Tps& tps1, const ZLIB::Tps& tps2) : "
         << "dimension is not even number \n";
    assert(0);
  }

  ZLIB::Tps result(0.0, 0);

  unsigned int i1, i2;
  for(unsigned int i=0; i < nd; i++){
    i2 = 2*i + 1;
    i1 = i2  - 1;
    result += D(tps1, i1)*D(tps2, i2);
    result -= D(tps1, i2)*D(tps2, i1);
  }
  return result;
}

// Friend functions

ZLIB::Tps ZLIB::sqrt(const ZLIB::Tps& zs)
{
  ZLIB::Tps el(zs), sum(0.0, 0), term(0.0, 0);
  
  double el0 = el[0];
  if(fabs(el0) < ZLIB_TINY)
  {
    cerr << "Error: sqrt(const ZLIB::Tps& zs)  ";
    cerr << "fabs(zs[0])"  << " <  " << ZLIB_TINY << "\n";
    assert(fabs(el0) >= ZLIB_TINY);
  }

  el -= el0;
  el *= -0.5/el0;;

  sum  = 1.;
  sum -= el;

  term = -el;

  unsigned int maxOrder = el.mltOrder();

  for(unsigned int i=2; i <= maxOrder; i++){
    term *= el;
    term *= (2*i - 3.)/i;
    sum += term;
  }

  sum *= std::sqrt(el0);

  return sum;
}

// Auxiliary methods

void ZLIB::Tps::initialize(double c, unsigned int o)
{
  checkOrder(o);

  _tps = new double[nmo(o)];
  if(!_tps){
    cerr << "Error : ZLIB::Tps::initialize(double c, o) : allocation failure \n";
    assert(_tps);
  }

  _tps[0] = c;
  for(unsigned int i=1; i < nmo(o); i++) _tps[i] = 0.0;

  _order = o;
}

void ZLIB::Tps::initialize(const ZLIB::Tps& zs)
{
  unsigned int zs_s = zs.size();

  _order  = zs._order;
  _tps = new double[zs_s];

  if(!_tps){
    cerr << "Error: ZLIB::Tps::initialize(const ZLIB::Tps& zs) : allocation failure \n";
    assert(_tps);
  }

  for(unsigned int i=0; i < zs_s; i++) _tps[i] = zs._tps[i];
}

void ZLIB::Tps::erase()
{
  if(_tps) delete [] _tps;
}

void ZLIB::Tps::checkOrder(unsigned int o) const
{
  if(o > maxOrder()) {
    cerr << "Error: ZLIB::Tps::checkOrder(unsigned int o) : "
         << "o(" << o << ") > maxOrder(" << maxOrder() << ") \n";
    assert(o <= maxOrder());
  }
}

void ZLIB::Tps::checkIndex(unsigned int index) const 
{
  if(index >= size()){
    cerr << "Error : ZLIB::Tps::checkIndex(unsigned int index) : " 
         << "index (" << index << ") > size (" << size() << ")\n";
    assert(index < size());
  }
}





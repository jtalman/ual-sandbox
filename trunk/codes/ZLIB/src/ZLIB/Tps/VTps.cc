// Library     : ZLIB
// File        : ZLIB/Tps/VTps.cc
// Copyright   : see Copyright file
// Author      : Yiton Yan
// C++ version : Nikolay Malitsky & Alexander Reshetov

#include "ZLIB/Tps/VTps.hh"

// Constructors

ZLIB::VTps::VTps(unsigned int s) : ZLIB::Space()  { initialize(s); }
ZLIB::VTps::VTps(const ZLIB::VTps& rhs) : ZLIB::Space() { initialize(rhs); }

// Destructor

ZLIB::VTps::~VTps() { erase(); }

// Attributes

unsigned int ZLIB::VTps::order() const
{
  unsigned int o = 0;
  for(unsigned int i = 0; i < _size; i++) o = max(o, _vtps[i].order());
  return o;
}

void ZLIB::VTps::order(unsigned int o)
{
  for(unsigned int i = 0; i < _size; i++)  _vtps[i].order(o);
}

void ZLIB::VTps::size(unsigned int s)
{
  if(_size == s) return;

  ZLIB::VTps tmp(s);

  unsigned int ms = min(_size, s);
  for(unsigned int i=0; i < ms; i++) tmp._vtps[i] = _vtps[i];

  erase();
  initialize(tmp);
}

// Assignment operators

ZLIB::VTps& ZLIB::VTps::operator=(double c)
{
  ZLIB::Tps tmp(0.0, 1);
  for(unsigned int i=0; i < _size; i++){
    _vtps[i] = tmp;
    _vtps[i]._tps[i+1] = c;
  }
  return *this;
}

ZLIB::VTps& ZLIB::VTps::operator+=(double c)
{
  for(unsigned int i=0; i < _size; i++){
    if(!_vtps[i].order()) _vtps[i].order(1);
    _vtps[i]._tps[i+1] += c;
  }
  return *this;
}

ZLIB::VTps& ZLIB::VTps::operator-=(double c) 
{ 
  return *this += -c; 
}

ZLIB::VTps& ZLIB::VTps::operator*=(double c)
{
  for(unsigned int i=0; i < _size; i++) _vtps[i] *= c;
  return *this;
}

ZLIB::VTps& ZLIB::VTps::operator/=(double c)
{
  return *this *= 1./c;
}

ZLIB::VTps& ZLIB::VTps::operator=(const ZLIB::VTps& rhs)
{
  if(this != &rhs) erase();
  initialize(rhs);
  return *this;
}

ZLIB::VTps& ZLIB::VTps::operator+=(const ZLIB::VTps& rhs)
{
  compareSize(rhs.size());
  for(unsigned int i=0; i < _size; i++) _vtps[i] += rhs._vtps[i];
  return *this;
}

ZLIB::VTps& ZLIB::VTps::operator-=(const ZLIB::VTps& rhs)
{
  compareSize(rhs.size());
  for(unsigned int i=0; i < _size; i++) _vtps[i] -= rhs._vtps[i];
  return *this;
}

ZLIB::VTps& ZLIB::VTps::operator*=(const ZLIB::VTps& rhs)
{
  return *this = (*this) * rhs;
}

ZLIB::VTps& ZLIB::VTps::operator/=(const ZLIB::VTps& rhs)
{
  return *this *= 1./rhs;
}

// Additive & Multiplicative Operators

ZLIB::VTps ZLIB::VTps::operator+(const ZLIB::VTps& rhs)
{
  return ZLIB::VTps(*this) += rhs;
}

ZLIB::VTps ZLIB::VTps::operator-(const ZLIB::VTps& rhs)
{
  return ZLIB::VTps(*this) -= rhs;
}

ZLIB::VTps ZLIB::VTps::operator*(const ZLIB::VTps& zvs)
{
  compareSize(zvs.size());

  ZLIB::Tps wk;
  ZLIB::VTps result(_size);

  // Concatenate for 0th and 1st orders

  for(unsigned int i=0; i < _size; i++){
    wk  = _vtps[0];
    wk *= zvs._vtps[i]._tps[1];
    wk += zvs._vtps[i]._tps[0];

    result._vtps[i] = wk;
  }

  for(unsigned int ii=1; ii < _size; ii++)
    for(unsigned int i=0; i < _size; i++){
      wk  = _vtps[ii];
      wk *= zvs._vtps[i]._tps[ii+1];
      
      result._vtps[i] += wk;
    } 

  // Concatenate for 2nd and higher orders

  _table->pntcct(zvs.order());

  int o = order();

  int mltO = mltOrder();
  mltOrder(min(mltO, max(o, zvs.order())));

  ZLIB::VTps tmp = *this;
  tmp.order(o);

  int j, iv, ivp;
  int nmcct2 = nmo(zvs.order()) - _size - 1;

  for(int jj = 1; jj <= nmcct2; jj++){
    j   = _table->jpc[jj] - 1;
    iv  = _table->ivpc[jj] - 1;
    ivp = _table->ivppc[jj] - 1;
    tmp._vtps[iv] = tmp._vtps[ivp]*_vtps[iv];
    for(unsigned int i=0; i < _size; i++) {
      result._vtps[i] += zvs.vtps(i,j)*tmp._vtps[iv];
    }
  }
  mltOrder(mltO);
  return result;
}

ZLIB::VTps ZLIB::VTps::operator/(const ZLIB::VTps& rhs)
{
  return (*this) * (1./rhs);
}


void ZLIB::VTps::propagate(ZLIB::Vector& p) const
{
  if(p.size() != dimension()){
    cerr << "Error: void ZLIB::VTps::propagate(ZLIB::Vector& p) :"
         << " p's size(" << p.size() << ") != "
         << " tps's dimension(" << dimension() << ") \n";
    assert(p.size() == dimension());
  }

  int mindim  = min(_size, p.size());
  int mino    = order();
  int minsize = nmo(mino);

  ZLIB::VTps tmp(_size);
  tmp = 1.;
  tmp._vtps[0].order(mino);  
  tmp.vtps(0,0) = 1.;

  int i, j;
  for(j = 2; j <= minsize; j++){
    int iv = _table->ivp[j];
    int jp = _table->jpp[j];
    tmp.vtps(0, j-1) = p[iv-1] * tmp.vtps(0, jp-1);
  }

  for(i=0; i < mindim; i++)  p[i] = vtps(i, 0);

  for(i=0; i < mindim; i++)
    for(j=1; j < minsize; j++)
      p[i] += vtps(i,j) * tmp.vtps(0,j);

 return;
}

// Friends

ZLIB::VTps ZLIB::operator-(const ZLIB::VTps& rhs) { return ZLIB::VTps(rhs) *= -1.; }

ZLIB::VTps ZLIB::operator+(const ZLIB::VTps& rhs, double c) { return ZLIB::VTps(rhs) += c; }
ZLIB::VTps ZLIB::operator+(double c, const ZLIB::VTps& rhs) { return rhs + c; }
ZLIB::VTps ZLIB::operator-(const ZLIB::VTps& rhs, double c) { return rhs + (-c); }
ZLIB::VTps ZLIB::operator-(double c, const ZLIB::VTps& rhs) { return -(rhs - c); }
ZLIB::VTps ZLIB::operator*(const ZLIB::VTps& rhs, double c) { return ZLIB::VTps(rhs) *= c; }
ZLIB::VTps ZLIB::operator*(double c, const ZLIB::VTps& rhs) { return rhs*c; }
ZLIB::VTps ZLIB::operator/(const ZLIB::VTps& rhs, double c) { return rhs * (1./c); }
ZLIB::VTps ZLIB::operator/(double , const ZLIB::VTps& )
{
  cerr << "Error: operator/(double c, const ZLIB::VTps& rhs) "
       << "is not implemened in this version \n";
  assert(0);
  return ZLIB::VTps();
}

// Dif. operators

ZLIB::VTps ZLIB::D(const ZLIB::VTps& vtps, int iv)
{
  ZLIB::VTps result(vtps.size());
  for(unsigned int i=0; i < vtps.size(); i++) result[i] = ZLIB::D(vtps[i], iv);
  return result;
}

ZLIB::VTps ZLIB::poisson(const ZLIB::Tps& tps, const ZLIB::VTps& vtps) 
{
  ZLIB::VTps result(vtps.size());
  for(unsigned int i=0; i < vtps.size(); i++) result[i] = ZLIB::poisson(tps, vtps[i]);
  return result;
}

ZLIB::VTps ZLIB::poisson(const ZLIB::VTps& vtps, const ZLIB::Tps& tps) 
{
  ZLIB::VTps result(vtps.size());
  for(unsigned int i=0; i < vtps.size(); i++) result[i] = ZLIB::poisson(vtps[i], tps);
  return result;
}

// I/0

void ZLIB::VTps::read(const char* f)
{
  ifstream file;
  file.open(f);
  if(!file) {
    cerr << "Cannot open " << f << " for input \n";
    assert(f);
  }
  file >> *this;
  file.close();
}

void ZLIB::VTps::write(const char* f)
{
  ofstream file;
  file.open(f);
  if(!file) {
    cerr << "Cannot open " << f << " for output \n";
    assert(file);
  }
  file << *this;
  file.close();
}

ostream& ZLIB::operator<<(ostream& out, const ZLIB::VTps& zvs)
{
 int dim   = zvs.dimension();
 int order = zvs.order();
 int size  = zvs.size();

 char s[80];

 out << "ZLIB::VTps : size = " << size ;
 out << " (dimension = " << dim ;
 out << "  order = " << order << " )\n\n";

 int nm1 = 0, nm2;
 for(int io = -1; io < order; io++){
   if(io > -1) nm1 = zvs.nmo(io);
   nm2 = zvs.nmo(io + 1);
   for(int i = nm1; i < nm2; i++){
       sprintf(s, "%5d ", i );
       out << s ;
       int j;
       for(j=0; j < size; j++)
       {
	 if((int) zvs._vtps[j].order() > io) sprintf(s, "% 19.13e ", zvs.vtps(j, i));
	 else sprintf(s, "% 19.13e ", 0.0);
	 out << s ;
       } 
       for(j=1; j <= dim; j++)
       {
          sprintf(s, "%3d", zvs._table->jv[j][i+1]);
          out << s ;
       } 
       out << "\n"; 
   }      
 }

 return(out);
}

istream& ZLIB::operator>>(istream& in, ZLIB::VTps& zvs)
{
  char s[80];

  int size, dim, order, nmo;

  in >> s >> s >> s >> s >> s;
  sscanf(s, "%d", &size);
//  size = atoi(s);
  in >> s >> s >> s;
  sscanf(s, "%d", &dim);
//  dim = atoi(s);
  in >> s >> s >> s;
  sscanf(s, "%d", &order);
//  order = atoi(s);
  nmo   = zvs.nmo(order);
  
  if(dim != (int) zvs.dimension()) {
    cerr << "Error: operator>>(ostream& in, ZLIB::VTps& zvs) : "
         << "Dimension of map to be read is not equal to dimension of zvs" << endl;
    assert(0);
  }

  if(order < 0 || order >  (int) zvs.maxOrder()) {
    cerr << "Error: Order of map (" << order << ") is out of [0," << zvs.maxOrder() << "] \n";
    assert(0);
  }

  if(size  != (int) zvs.size())  zvs.size(size);
  zvs.order(order);

  in >> s;
  for(int i=0; i < nmo; i++){
    in >> s;
    int j;
    for(j=0; j < dim; j++) in >> zvs.vtps(j, i);
    for(j=0; j < dim; j++) in >> s;
  }

  return(in);

}

// Private methods

void ZLIB::VTps::initialize(unsigned int s)
{ 
  _vtps = 0;
  _size = 0;

  if(!s) return;

  if(s > dimension()) {
    cerr << "Error : ZLIB::VTps::initialize(unsigned int s) : "
         << "s(" << s << ") > dimension(" << dimension() << ") \n";
    assert(0);
  }

  _vtps = new ZLIB::Tps[s];
  if(!_vtps){
    cerr << "Error: ZLIB::VTps::initialize(unsigned int s) : allocation failure \n";
    assert(_vtps);
  }  
  _size = s;
}

void ZLIB::VTps::initialize(const ZLIB::VTps& zvs)
{
  _vtps = 0;
  _size = zvs.size(); 

  if(!_size) return;

  _vtps = new ZLIB::Tps[_size];
  if(!_vtps){
    cerr << "Error: ZLIB::VTps::initialize(const ZLIB::VTps& zvs) : allocation failure \n";
    assert(_vtps);
  } 

  for(unsigned int i=0; i < _size; i++) _vtps[i] = zvs._vtps[i];

}

void ZLIB::VTps::erase()
{
  if(_vtps) delete [] _vtps;
  _vtps = 0;
  _size = 0;
}

void ZLIB::VTps::compareSize(unsigned int s) const 
{
  if(_size != s){
    cerr << "Error: ZLIB::VTps::compareSize(int s) : "
         << "s (" << s << ") != _size(" << _size << ") \n";
    assert(_size == s);
  }

}

void ZLIB::VTps::checkIndex(unsigned int index) const
{
  if(index >= _size){
    cerr << "Error : ZLIB::VTps::checkIndex(unsigned int index) : "
         << "index(" << index << ") > _size(" << _size << ") \n";
    assert(index < _size);
  }
}







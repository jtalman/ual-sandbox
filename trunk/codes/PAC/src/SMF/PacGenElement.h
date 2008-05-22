// Library     : PAC
// File        : SMF/PacGenElement.h
// Copyright   : see Copyright file
// Description : PacGenElement is a basis class for all element types,such
//               as PacElement, PacDrift etc. Collection of PacGenElement's
//               composes the first level of Standard Machine Format.
// Author      : Nikolay Malitsky

#ifndef PAC_GEN_ELEMENT_H
#define PAC_GEN_ELEMENT_H

#include "Templates/PacNamedPtr.h"

#include "Optics/PacVTps.h"
#include "SMF/PacElemKey.h"
#include "SMF/PacElemPart.h"

class PacGenElement 
{
public:

// Constructors & copy operator

   PacGenElement(const string& name = string(""), int key = 0);
   PacGenElement(const PacGenElement& e) : _ptr(e._ptr) {}

   PacGenElement& operator = (const PacGenElement& e) { checkName(); _ptr = e._ptr; return *this;} 

// User Interface

   PacVTps& map()              { return _ptr->_map; }
   const PacVTps& map() const  { return _ptr->_map; }
   void map(const PacVTps& m)  { _ptr->_map = m; }

   PacElemPart& body()     { return *setBody(); }
   PacElemPart& front()    { return *setFront(); }
   PacElemPart& end()      { return *setEnd(); }

   void set(const PacElemAttributes& att)  { setBody()->set(att); }
   void set(const PacElemBucket& bucket)   { setBody()->set(bucket);}

   void add(const PacElemAttributes& att)  { setBody()->add(att); }
   void add(const PacElemBucket& bucket)   { setBody()->add(bucket);}  

   double get(const PacElemAttribKey& key) const { return (getBody() != 0 ? getBody()->get(key) : 0.0); }

   void remove(const PacElemAttribKey& key){ setBody()->remove(key); }
   void remove()                           { setBody()->remove(); }

   PacElemAttributes& rms()  { return setBody()->rms(); }

// Name & Count

   const string& name() const { return _ptr.name(); }
   int count() const { return _ptr.count(); }

// Data

   int  key() const { return _ptr->_key; }

   const string& type() const;

   PacElemPart* getBody() const { return _ptr->_parts[1]; }
   PacElemPart* setBody() { return create(1); }

   PacElemPart* getFront() const { return _ptr->_parts[0]; }
   PacElemPart* setFront() { return create(0); }

   PacElemPart* getEnd() const { return _ptr->_parts[2]; }
   PacElemPart* setEnd() { return create(2); }

   PacElemPart* getPart(int i) const { checkPart(i); return _ptr->_parts[i]; }
   PacElemPart* setPart(int i) { checkPart(i); return create(i); }

// Interface to global items

   PacGenElement* operator()(const string& name) const;

// Friends

   friend class PacNameOfGenElement;

protected:

   class Data
   {
   public:

     Data() : _key(0) { for(int i=0; i < 3; i++) _parts[i] = 0; }
    ~Data() { for(int i=0; i < 3; i++) if(_parts[i]) delete _parts[i]; }
    
     int _key;

     PacVTps _map;

     PacElemPart* _parts[3];

   };

   typedef PacNamedPtr<Data> smart_pointer;
   smart_pointer _ptr;

protected:

   void checkName();
   void checkPart(int i) const;
   PacElemPart* create(int n);

};

struct PacNameOfGenElement
{
  const string& operator()(const PacGenElement& x) const 
  { return x.name(); }

  void operator()(PacGenElement& x, const string& key) const 
  { x._ptr = PacNamedPtr<PacGenElement::Data>(key); }

  int  count(const PacGenElement& x) const 
  { return x._ptr.count(); }
};


template<int kkey>  
class PacGenElemType: public PacGenElement                                           
{                                                                                        
public:                                                                                  

// Constructors & destructor

  PacGenElemType() : PacGenElement() {}
  PacGenElemType(const string& name) : PacGenElement(name, kkey) {}
  PacGenElemType(const PacGenElement& e) : PacGenElement(e) {}

// Copy operator

   PacGenElement& operator = (const PacGenElement& e) 
   { PacGenElement::operator=(e); return *this;} 

};

       
#define PacRbend       PacGenElemType<1>
#define PacSbend       PacGenElemType<2>
#define PacQuadrupole  PacGenElemType<3>
#define PacSextupole   PacGenElemType<4>
#define PacOctupole    PacGenElemType<5>
#define PacMultipole   PacGenElemType<6>
#define PacSolenoid    PacGenElemType<7>
#define PacHmonitor    PacGenElemType<8>
#define PacVmonitor    PacGenElemType<9>
#define PacMonitor     PacGenElemType<10>
#define PacHkicker     PacGenElemType<11>
#define PacVkicker     PacGenElemType<12>
#define PacKicker      PacGenElemType<13>
#define PacRfCavity    PacGenElemType<14>
#define PacElSeparator PacGenElemType<15>
#define PacEcollimator PacGenElemType<16>
#define PacRcollimator PacGenElemType<17>
#define PacYrot        PacGenElemType<18>
#define PacSrot        PacGenElemType<19>
#define PacInstrument  PacGenElemType<20>
#define PacBeamBeam    PacGenElemType<21>
#define PacDrift       PacGenElemType<97>
#define PacMarker      PacGenElemType<98>
#define PacElement     PacGenElemType<99>

#endif

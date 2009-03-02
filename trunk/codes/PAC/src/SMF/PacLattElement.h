// Library     : PAC
// File        : SMF/PacLattElement.h
// Copyright   : see Copyright file
// Description : 
// Author      : Nikolay Malitsky

#ifndef PAC_LATT_ELEMENT_H
#define PAC_LATT_ELEMENT_H

#include "UAL/SMF/AcceleratorNode.hh"
#include "Templates/PacRCIPtr.h"
#include "SMF/PacGenElement.h"

class PacLattice;

class PacLattElement : public UAL::AcceleratorNode
{
public:

// Constructors & copy operator

   PacLattElement() : _ptr(new PacLattElement::Data()) {}
   PacLattElement(const PacGenElement& e) : _ptr(new PacLattElement::Data()) { define(e); }
   PacLattElement(const PacLattElement& e) : _ptr(e._ptr) {}

   void operator = (const PacLattElement& e) { checkGenElement(); _ptr = e._ptr; }
   void operator = (const PacGenElement& e)  { checkGenElement(); define(e);} 

   // Accelerator Node interface

   /** Returns a type */
   const std::string& getType() const { return type(); }

   /** Returns a name */
   const std::string& getName() const { return name(); }

   /** Returns a design name */
   const std::string& getDesignName() const { return genElement().name(); } 

   /** Redefine a length */
   void addLength(double l);

   /** Returns a length */
   double getLength() const;

   double getN() const;

   void addN(double n);
   
   /** Redefine an angle */
   void addAngle(double angle);

   /** Returns an angle */
   double getAngle() const;

   /** Returns a longitudinal position */
   double getPosition() const;

   /** Sets a longitudinal position */
   void setPosition(double at);   


// User Interface

   PacVTps& map()              { return _ptr->_map; }
   const PacVTps& map() const  { return _ptr->_map; }
   void map(const PacVTps& m)  { _ptr->_map = m; }

   PacElemAttributes& body()    { return *setBody(); }
   PacElemAttributes& front()   { return *setFront(); } 
   PacElemAttributes& end()     { return *setEnd(); }

   void set(const PacElemAttributes& att)  { setBody()->set(att); }
   void set(const PacElemBucket& bucket)   { setBody()->set(bucket);}

   void add(const PacElemAttributes& att)  { setBody()->add(att); }
   void add(const PacElemBucket& bucket)   { setBody()->add(bucket);}  

   double get(const PacElemAttribKey& key) const { return (getBody() != 0 ? getBody()->get(key) : 0.0); }

   void remove(const PacElemAttribKey& key) { setBody()->remove(key); }
   void remove() { setBody()->remove(); }

// Name & Count

   string& name()              { return _ptr->_name; }
   const string& name() const  { return _ptr->_name; }
   void name(const string& n)  {_ptr->_name = n; }

   int key() const { return _ptr->_key; }
   void key(int k) { _ptr->_key = k; }

   const string& type() const;

   int count() const { return _ptr.count(); }

// Data

   const PacGenElement& genElement() const { return _ptr->_genElement; }
   PacGenElement& genElement() { return _ptr->_genElement; }


   PacElemAttributes* getBody() const { return _ptr->_parts[1]; }
   PacElemAttributes* setBody()       { return create(1); }

   PacElemAttributes* getFront() const { return _ptr->_parts[0]; }
   PacElemAttributes* setFront()       { return create(0); }

   PacElemAttributes* getEnd() const { return _ptr->_parts[2]; }
   PacElemAttributes* setEnd()       { return create(2); }

   PacElemAttributes* getPart(int i) const { checkPart(i); return _ptr->_parts[i]; }
   PacElemAttributes* setPart(int i)       { checkPart(i); return create(i); }

   // Friend

   friend class PacLattice;

protected:

   class Data
   {
   public:

     Data() { _at = 0.0; for(int i=0; i < 3; i++) _parts[i] = 0; }
    ~Data() { for(int i=0; i < 3; i++) if(_parts[i]) delete _parts[i]; }
    
     string _name;
     int    _key;

     double _at;

     PacVTps       _map;
     PacElemAttributes* _parts[3];

     PacGenElement _genElement;

   };

   typedef PacRCIPtr<Data> smart_pointer;
   smart_pointer _ptr;

protected:

   void checkGenElement();

   void check(const PacGenElement& e);
   void define(const PacGenElement& e);

   void checkPart(int i) const;
   PacElemAttributes* create(int n);

};


#endif

// Library     : PAC
// File        : SMF/PacLattice.h
// Copyright   : see Copyright file
// Description : PacLattice represents a "flat" or "fully-instantiated" view of accelerator 
//               sector. It is implemented as a vector of PacLattElement's. The collection 
//               of PacLattice's composes the third level of Standard Machine Format. 
// Author      : Nikolay Malitsky

#ifndef PAC_LATTICE_H
#define PAC_LATTICE_H

#include "UAL/SMF/AcceleratorNode.hh"

#include "SMF/PacLine.h"
#include "SMF/PacLattElement.h"

#include "Templates/PacVector.h"

class PacLattice : public UAL::AcceleratorNode
{
public:

// Constructors & copy operator

   PacLattice();
   PacLattice(const string& name);
   PacLattice(const PacLattice& rhs);

   void operator  = (const PacLattice& rhs);

   // Accelerator Node interface

   /** Returns the name */
   const std::string& getName() const { return name(); }

   /** Returns the number of children */
   int getNodeCount() const { return size(); }

   /** Returns the node child  */   
   UAL::AcceleratorNode* const  getNodeAt(int indx) const { return &(_ptr->_vector[indx]);}

   /** Returns a deep copy of this node */
   virtual UAL::AcceleratorNode* clone() const { return new PacLattice(*this); }
   

// Modifiers

   void set(PacList<PacLattElement>& array);
   void set(const PacLattice& la);
   void set(PacLine& li); 
   void add(const PacLattice& la);
   void erase();

   friend PacLattice operator , (const PacLattice& l1, const PacLattice& l2);

// Access methods

   const string& name() const { return _ptr.name(); }; 

// ... to lattice elements

   int size() const;
   PacLattElement& operator[](int index);
   const  PacLattElement& operator[](int index) const;
   PacVector<int> indexes(const char* name);

// ... to line

   const string& line() const;

// ... to lattices

   typedef PacList<string>   list_type;
   typedef list_type::iterator       iterator;
   typedef list_type::const_iterator const_iterator;

   const PacList<string>& lattices() const;

// ... to state

   int count() const ;
   int isLocked() const;
   void lock();

// ... to collection items

   PacLattice* operator()(const string& name) const;

// Friends

   friend class PacNameOfLattice;

protected:

class Data
{
public:

  Data() : _locker(0), _size(0), _vector(0) {}
 ~Data() { if(_vector) delete [] _vector; }

  int _locker;

  int _size;
  PacLattElement* _vector;

  string _line;
  PacList<string> _lattices;

};

   typedef PacNamedPtr<Data> smart_pointer;
   smart_pointer _ptr;

private:

// Lattice

     void check();
     void checkName();

     void setLattice(const PacLattice& l1, const PacLattice& l2);
     void setLattice(PacLine& li);
     void addLattice(const PacLattice& l1);
     void eraseLattice();


// Vector

     void setVector(const PacLattice& l1, const PacLattice& l2);
     void setVector(const PacLine& li);

// Line

     void check(const PacLine& li);
     int  count(const PacLine& li);
     void track(const PacLine& li, int direction, int& counter);
};

// 

struct PacNameOfLattice
{
  const string& operator()(const PacLattice& x) const 
  { return x.name(); }

  void operator()(PacLattice& x, const string& key) const 
  { x._ptr = PacNamedPtr<PacLattice::Data>(key); }

  int  count(const PacLattice& x) const 
  { return x._ptr.count(); }
};


// Access methods

// ... to lattice elements

inline int  PacLattice::size() const { return _ptr->_size; }
inline PacLattElement& PacLattice::operator[](int index) { return _ptr->_vector[index]; }
inline const PacLattElement& PacLattice::operator[](int index) const { return _ptr->_vector[index]; }

// ... to line

inline const string& PacLattice::line() const { return _ptr->_line; }

// ... to lattices

inline const PacList<string>& PacLattice::lattices() const { return _ptr->_lattices; }

// ... to state

inline int  PacLattice::count() const { return _ptr.count(); }
inline int  PacLattice::isLocked() const { return _ptr->_locker; }
inline void PacLattice::lock() { _ptr->_locker = 1; }


#endif

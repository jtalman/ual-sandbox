// Library     : PAC
// File        : SMF/PacLine.h
// Copyright   : see Copyright file
// Description : PacLine implements a beamline of elements (a tree of PacLine's
//               and PacGenElement's). The collection of them composes the second 
//               level of Standard Machine Format. 
// Author      : Nikolay Malitsky

#ifndef PAC_LINE_H
#define PAC_LINE_H

#include "SMF/PacGenElement.h"
#include "Templates/PacList.h"

class PacLine;

class PacLineNode 
{
public:

  PacLineNode();
  PacLineNode(const PacLine& l, int p = 1);
  PacLineNode(const PacGenElement& e, int p = 1);
  PacLineNode(const PacLineNode& n);
 ~PacLineNode();

  void operator=(const PacLineNode& l);

  int repetition() const { return _repetition; }
  const PacGenElement& element() const { return _element; }
  const PacLine& line() const;

private:

  // data
  int _repetition;
  PacGenElement _element;

  // link
  PacLine*       _line;

  void create();
  void define(const PacLineNode& n);
  void check(const PacGenElement& e);

};


class PacLine
{
public:

// Constructors & copy operator

   PacLine(const string& name = string(""));
   PacLine(const PacLine& l) : _ptr(l._ptr) {}

   void operator = (const PacLine& l) { checkName();  _ptr = l._ptr; }

// User Interface

   void set(const PacLine& l) { erase(); push_back(l);}
   void set(const PacGenElement& e) { erase(); push_back(e);} 

   void add(const PacLine& l) { push_back(l); }
   void add(const PacGenElement& e) { push_back(e); } 
 
   void erase();

   friend PacLine& operator , (const PacLine& l1, const PacLine& l2);
   friend PacLine& operator , (const PacGenElement& e, const PacLine& l);
   friend PacLine& operator , (const PacLine& l, const PacGenElement& e);
   friend PacLine& operator , (const PacGenElement& e1, const PacGenElement& e2);

   friend PacLine operator*(int p, PacGenElement& e);
   friend PacLine operator*(int p, PacLine& l);

// Name & Count

   const string& name() const { return _ptr.name(); }
   int count() const { return _ptr.count(); }

// Data

   typedef PacList<PacLineNode>                list_type;
   typedef list_type::iterator                 iterator;
   typedef list_type::const_iterator           const_iterator;
   typedef list_type::reverse_iterator         reverse_iterator;
   typedef list_type::const_reverse_iterator   const_reverse_iterator;

   const list_type& list()  const  { return (count() != 0 ? _ptr->_list : _empty_list); }

   int isLocked() const { return ( count() != 0 ? _ptr->_locker : 0); }
   void lock() { if(count()) _ptr->_locker = 1; }

// Interface  to collection items

   PacLine* operator()(const string& name) const;

// Friends

   friend class PacLineNode;
   friend class PacNameOfLine;

protected:

class Data
{
public:

  Data() : _locker(0) {}
  int _locker;
  PacList<PacLineNode> _list;

};
   static PacList<PacLineNode> _empty_list;   
   typedef PacNamedPtr<Data> smart_pointer;
   smart_pointer _ptr;

protected:

// Auxiliary methods

     void check();
     void checkName();
     void checkTmpLine(const PacLine& l1, const PacLine& l2);

     void push_back(const PacLine& l);
     void push_back(const PacGenElement& e);

     void create();

};


struct PacNameOfLine
{
  const string& operator()(const PacLine& x) const 
  { return x.name(); }

  void operator()(PacLine& x, const string& key) const 
  { x._ptr = PacNamedPtr<PacLine::Data>(key); }

  int  count(const PacLine& x) const 
  { return x._ptr.count(); }
};

#endif

// #include <iostream.h>
#include "sxf/SequenceStack.hh"

// Constructor.
SXF::SequenceStack::SequenceStack(int size) 
  :  m_iSize(size), m_iNextSlot(0)
{
  if(size <= 0) {
    cout << "Error : SXF::SequenceStack constructor : "
	 << "size(" << size << ") <= 0 " << endl;
  }
  m_pSequences = new SXF::Sequence*[size];
  for(int i=0; i < m_iSize; i++) { m_pSequences[i] = 0; }
}

// Destructor.
SXF::SequenceStack::~SequenceStack()
{
  if(m_pSequences) { delete [] m_pSequences; }
}

// Return the max stack size.
int SXF::SequenceStack::size() const 
{
  return m_iSize;
}

// Return true if stack is empty.
int SXF::SequenceStack::isEmpty() const
{
  return m_iNextSlot == 0;
}

// Return and remove the topmost element in the stack.
SXF::Sequence* SXF::SequenceStack::pop()
{
  if(isEmpty()){
    cout << "Error : SXF::SequenceStack pop : "
	 << "attempt to access the empty stack" << endl;
  }
  return m_pSequences[--m_iNextSlot];
}


// Return the topmost element in the stack.
SXF::Sequence* SXF::SequenceStack::top()
{
  if(isEmpty()){
    cout << "Error : SXF::SequenceStack top : "
	 << "attempt to access the empty stack" << endl;
  }
  return m_pSequences[m_iNextSlot - 1];
}

void SXF::SequenceStack::push(SXF::Sequence* val)
{
  if(m_iNextSlot >= size()) {
    cout << "Error : SXF::SequenceStack push : "
	 << "stack(" << size() << ") is full" << endl;
  }  
  m_pSequences[m_iNextSlot++] = val;  
}

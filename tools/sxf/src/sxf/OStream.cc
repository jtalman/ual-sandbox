#include "sxf/OStream.hh"

// Constructor.
SXF::OStream::OStream(ostream& out)
  : m_refOStream(out), 
    m_iLineNumber(0),
    m_iCFECounter(0),  
    m_iSyntaxCounter(0) 
{
}
 
// Return an output stream + increment a CFE (front end) error counter.
ostream& SXF::OStream::cfe_error()
{
  m_iCFECounter++;
  return  m_refOStream;
}

// Return an output stream + increment an syntax error counter.
ostream& SXF::OStream::syntax_error()
{
  m_iSyntaxCounter++;
  return  m_refOStream;
}

// Set a line number.
int SXF::OStream::set_lineno()
{
  return m_iLineNumber = 1;
}

// Increment a line number.
int  SXF::OStream::increment_lineno()
{
  return ++m_iLineNumber;
}

// Write a line.
void SXF::OStream::write_line(const char* line)
{
  m_refOStream << line;
}

// Write a status.
void SXF::OStream::write_status()
{
  if(m_iCFECounter == 0 &&  m_iSyntaxCounter == 0) return;

  m_refOStream << "\n\n";
  m_refOStream << "********************************************" << endl;
  m_refOStream << "Syntax Errors: " << m_iSyntaxCounter << endl;
  m_refOStream << "CFE    Errors: " << m_iCFECounter << endl;
  m_refOStream << "********************************************" << endl;

}

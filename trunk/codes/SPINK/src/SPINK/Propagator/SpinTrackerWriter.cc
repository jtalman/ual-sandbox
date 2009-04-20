
#include <iostream>
#include "SPINK/Propagator/SpinTrackerWriter.hh"


SPINK::SpinTrackerWriter* SPINK::SpinTrackerWriter::s_theInstance = 0;

SPINK::SpinTrackerWriter::SpinTrackerWriter()
{
}

SPINK::SpinTrackerWriter* SPINK::SpinTrackerWriter::getInstance()
{
  if(s_theInstance == 0) {
    s_theInstance = new SPINK::SpinTrackerWriter();
  }
  return s_theInstance;
}

void SPINK::SpinTrackerWriter::setFileName(const char* filename)
{
  m_fileName = filename;
}

void SPINK::SpinTrackerWriter::write(double t)
{
  //  std::cout << "File name " << m_fileName << " " << t << std::endl;
}



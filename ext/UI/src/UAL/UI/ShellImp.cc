
#include "UAL/UI/ShellImp.hh"

UAL::ShellImp* UAL::ShellImp::s_theInstance = 0;

UAL::ShellImp::ShellImp()
{
  m_space = 0;
}

UAL::ShellImp& UAL::ShellImp::getInstance()
{
  if(s_theInstance == 0) {
    s_theInstance = new UAL::ShellImp();
  }
  return *s_theInstance;
}

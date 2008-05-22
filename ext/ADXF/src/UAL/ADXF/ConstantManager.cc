#include "UAL/ADXF/ConstantManager.hh"

UAL::ADXFConstantManager* UAL::ADXFConstantManager::s_theInstance = 0;

UAL::ADXFConstantManager* UAL::ADXFConstantManager::getInstance()
{
  if(s_theInstance == 0){
    s_theInstance = new UAL::ADXFConstantManager();
  }
  return s_theInstance;
}

UAL::ADXFConstantManager::ADXFConstantManager()
{
}

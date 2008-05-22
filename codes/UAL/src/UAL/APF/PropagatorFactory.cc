// Library       : UAL
// File          : UAL/APF/PropagatorFactory.cc
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#include "UAL/APF/PropagatorFactory.hh"

UAL::PropagatorFactory* UAL::PropagatorFactory::s_theInstance = 0;

UAL::PropagatorFactory::PropagatorFactory()
{
}

UAL::PropagatorFactory& UAL::PropagatorFactory::getInstance()
{
  if(s_theInstance == 0){
    s_theInstance = new UAL::PropagatorFactory();
  }
  return *s_theInstance;
}

UAL::PropagatorNode* UAL::PropagatorFactory::createPropagator(const std::string& classname)
{
  UAL::PropagatorFactory::Iterator it = find(classname);
  if(it == end()) return 0;
  return it->second->clone();
}

void UAL::PropagatorFactory::add(const std::string& classname, 
				  const UAL::PropagatorNodePtr& ptr)
{
  UAL::PropagatorFactory::Iterator it = m_registry.find(classname);
  if(it == m_registry.end()){
    m_registry[classname] = ptr;
  }  
}

UAL::PropagatorFactory::Iterator UAL::PropagatorFactory::begin()
{
  return m_registry.begin();
}

UAL::PropagatorFactory::Iterator UAL::PropagatorFactory::find(const std::string& classname)
{
  return m_registry.find(classname);
}

UAL::PropagatorFactory::Iterator UAL::PropagatorFactory::end()
{
  return m_registry.end();
}

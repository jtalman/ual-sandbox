#include "Arguments.hh"

UAL::Arguments::Arguments()
{
}

UAL::Arguments::~Arguments()
{
  std::map<std::string, Argument*>::iterator it;
  for(it = m_arguments.begin(); it != m_arguments.end(); it++){
    delete it->second;
  }
}

UAL::Arguments::Arguments(const UAL::Argument& arg)
{
  m_arguments[arg.getKey()] = new UAL::Argument(arg);
}


UAL::Arguments& UAL::Arguments::operator << (const UAL::Argument& arg )
{
  m_arguments[arg.getKey()] = new UAL::Argument(arg);
  return *this;
}

UAL::Arguments& UAL::operator , (UAL::Arguments& args, const UAL::Argument& arg )
{
  args.m_arguments[arg.getKey()] = new UAL::Argument(arg);
  return args;
}



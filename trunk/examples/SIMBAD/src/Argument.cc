#include "Argument.hh"

UAL::Object UAL::Argument::s_emptyObject;

UAL::Argument::Argument()
  : m_type(UAL::Argument::EMPTY), m_o(s_emptyObject) 
{
}

UAL::Argument::Argument(const UAL::Argument& arg)
  : m_type(arg.m_type), m_key(arg.m_key), m_n(arg.m_n), m_s(arg.m_s), m_o(arg.m_o)
{
}

UAL::Argument::Argument(const std::string& key, int value)
  : m_type(UAL::Argument::NUMBER), m_key(key), m_n(value), m_o(s_emptyObject) 
{
}

UAL::Argument::Argument(const std::string& key, double value)
  : m_type(UAL::Argument::NUMBER), m_key(key), m_n(value), m_o(s_emptyObject) 
{
}

UAL::Argument::Argument(const std::string& key, const std::string& value)
  : m_type(UAL::Argument::STRING), m_key(key), m_s(value), m_o(s_emptyObject) 
{
}

UAL::Argument::Argument(const std::string& key, Object& value)
  : m_type(UAL::Argument::OBJECT), m_key(key), m_o(value) 
{
}

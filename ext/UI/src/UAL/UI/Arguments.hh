#ifndef UAL_ARGUMENTS_HH
#define UAL_ARGUMENTS_HH

#include <map>

#include "Argument.hh"

namespace UAL {

  /** Collection of arguments used in the Shell methods */

  class Arguments {

  public :

    /** Constructor */
    Arguments();

    /** Copy constructor */
    Arguments(const Argument& arg);

    /** Destructor */  
    ~Arguments();

    /** Adds argument */
    Arguments& operator << (const Argument& arg);

    /** Returns the STL map with arguments */
    const std::map<std::string, UAL::Argument*> getMap() const { return m_arguments; }

  private:

    std::map<std::string, UAL::Argument*> m_arguments;

  };

  typedef Arguments Args;

}



#endif

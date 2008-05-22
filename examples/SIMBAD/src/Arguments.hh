#ifndef UAL_ARGUMENTS_HH
#define UAL_ARGUMENTS_HH

#include <map>

#include "Argument.hh"

namespace UAL {

  class Arguments {

  public :

    /** Constructor */
    Arguments();

    /** Constructor */
    Arguments(const Argument& arg);

    /** Destructor */  
    ~Arguments();

    /** Add argument */
    Arguments& operator << (const Argument& arg);

    /** Add argument */
    friend Arguments& operator , (Arguments& args, const Argument& arg);

    /** Returns the STL map with arguments */
    const std::map<std::string, Argument*> getMap() const { return m_arguments; }

  private:

    std::map<std::string, Argument*> m_arguments;

  };

  typedef Arguments Args;

};



#endif

#ifndef UAL_ARGUMENT_HH
#define UAL_ARGUMENT_HH

#include <string>
#include "UAL/Common/Object.hh"

namespace UAL {

  /** Argument of the Shell methods */

  class Argument {

  public:

    /** Value types */
    enum Type {
      EMPTY  = -1,
      NUMBER = 0,
      STRING,
      OBJECT
    };

    /** Constructor */
    Argument();

    /** Copy Constructor */
    Argument(const Argument& arg);   

    /** Constructor for integer value  */
    Argument(const std::string& key, int value);

    /** Constructor for double value */
    Argument(const std::string& key, double value);

    /** Constructor for string value */
    Argument(const std::string& key, const std::string& value);

    /** Constructor for Object value */
    Argument(const std::string& key, Object& value);

    /** Returns value type */
    Type getType() const { return m_type; }

    /** Returns argument name */
    const std::string&  getKey() const { return m_key; }

    /** Returns a number */
    double getNumber() const { return m_n; }

    /** Returns a string value */
    const std::string&  getString() const { return m_s; }

    /** Returns an object value */
    Object&  getObject() const { return m_o; }

  private:

    Type m_type;

    std::string m_key;

    double      m_n;
    std::string m_s;
    Object&     m_o;

    static Object s_emptyObject;
    
  };

  typedef Argument Arg;

}

#endif 

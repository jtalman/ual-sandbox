//# Library     : UAL
//# File        : UAL/ADXF/ConstantManager.hh
//# Copyright   : see Copyrigh file


#ifndef UAL_ADXF_CONSTANT_MANAGER_HH
#define UAL_ADXF_CONSTANT_MANAGER_HH

#include <string>
#include <numeric> // for accumulate

#include "muParser.h"
#include "muParserInt.h"

namespace UAL {

  /**
   * The parser and manager of ADXF constants
   */ 

  class ADXFConstantManager 
  {
  public:

    static ADXFConstantManager* getInstance();

    mu::Parser muParser;

  protected:

    static ADXFConstantManager* s_theInstance;

  private:

    ADXFConstantManager();

  };

}

#endif

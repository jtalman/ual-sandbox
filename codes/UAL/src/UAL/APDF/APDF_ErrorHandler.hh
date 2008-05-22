// Library       : UAL
// File          : UAL/APDF/APDF_ErrorHandler.hh
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#ifndef UAL_APDF_ERROR_HANDLER_HH
#define UAL_APDF_ERROR_HANDLER_HH

#include <xercesc/dom/DOMErrorHandler.hpp>

using namespace xercesc;

namespace UAL {

  /** ErrorHandler of the APDF DOM Builder */

  class APDF_ErrorHandler : public DOMErrorHandler  {

  public:

    /** Constructor */
    APDF_ErrorHandler();

    /** Destructor */
    virtual ~APDF_ErrorHandler();

    /** The error handler interface */
    bool handleError(const DOMError& domError);

    void resetErrors();

  private:

    /* Unimplemented constructors and operators*/
    APDF_ErrorHandler(const APDF_ErrorHandler&);

    void operator=(const APDF_ErrorHandler&);


  };

}


#endif

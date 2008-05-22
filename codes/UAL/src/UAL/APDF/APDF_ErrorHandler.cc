// Library       : UAL
// File          : UAL/APF/APDF_ErrorHandler.cc
// Copyright     : see Copyright file
// Authors       : N.Malitsky & R.Talman

#include <iostream.h>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOMError.hpp>

#include "UAL/APDF/APDF_ErrorHandler.hh"

UAL::APDF_ErrorHandler::APDF_ErrorHandler()
{
}

UAL::APDF_ErrorHandler::APDF_ErrorHandler(const UAL::APDF_ErrorHandler&)
{
}


UAL::APDF_ErrorHandler::~APDF_ErrorHandler()
{
}

void UAL::APDF_ErrorHandler::operator=(const UAL::APDF_ErrorHandler&)
{
}

bool UAL::APDF_ErrorHandler::handleError(const DOMError &domError)
{
    // Display whatever error message passed from the serializer
    if (domError.getSeverity() == DOMError::DOM_SEVERITY_WARNING)
        std::cerr << "\nWarning Message: ";
    else if (domError.getSeverity() == DOMError::DOM_SEVERITY_ERROR)
        std::cerr << "\nError Message: ";
    else
        std::cerr << "\nFatal Message: ";

    char *msg = XMLString::transcode(domError.getMessage());
    std::cerr<< msg << std::endl;
    XMLString::release(&msg);

    // Instructs the serializer to continue serialization if possible.
    return true;
}

void UAL::APDF_ErrorHandler::resetErrors()
{
}


//A linkDef file is needed because of the namespace.

#ifdef __CINT__

//below is default for any class ROOT makes.
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//  these lines deal with the namespace
#pragma link C++ namespace ZLIB;
#pragma link C++ nestedclasses;

//Here are the classes and structs.
#pragma link C++ class ZLIB::Space-!;
#pragma link C++ class ZLIB::Vector-!;
#pragma link C++ class ZLIB::Tps-!;
#pragma link C++ class ZLIB::VTps-!;

//THese lines are needed since these functions are outside of classes.
#pragma link C++ function ZLIB::D(const Tps&, unsigned int);
#pragma link C++ function ZLIB::poisson(const Tps& , const Tps& );
#pragma link C++ function ZLIB::operator-(const Tps& );
#pragma link C++ function ZLIB::operator+(const Tps& , double );
#pragma link C++ function ZLIB::operator+(double , const Tps& );
#pragma link C++ function ZLIB::operator-(const Tps& , double );
#pragma link C++ function ZLIB::operator-(double , const Tps& );
#pragma link C++ function ZLIB::operator*(const Tps& , double );
#pragma link C++ function ZLIB::operator*(double , const Tps& );
#pragma link C++ function ZLIB::operator/(const Tps& , double );
#pragma link C++ function ZLIB::operator/(double , const Tps& );
#pragma link C++ function ZLIB::sqrt(const Tps&);
#pragma link C++ function ZLIB::D(const VTps& , int );  
#pragma link C++ function ZLIB::poisson(const Tps& , const VTps&);
#pragma link C++ function ZLIB::poisson(const VTps& , const Tps& );   
#pragma link C++ function ZLIB::operator-(const VTps& );
#pragma link C++ function ZLIB::operator+(const VTps& , double );
#pragma link C++ function ZLIB::operator+(double , const VTps& );
#pragma link C++ function ZLIB::operator-(const VTps& , double );
#pragma link C++ function ZLIB::operator-(double , const VTps& );
#pragma link C++ function ZLIB::operator*(const VTps& , double );
#pragma link C++ function ZLIB::operator*(double , const VTps& );
#pragma link C++ function ZLIB::operator/(const VTps& , double );
#pragma link C++ function ZLIB::operator/(double , const VTps& ); 

 
#endif

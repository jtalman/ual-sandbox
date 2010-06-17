#include <string>
#include <iostream>

#ifndef THINSPIN_FOUR_TENSOR_HH
#define THINSPIN_FOUR_TENSOR_HH

namespace THINSPIN {
   class fourTensor {
      public:
         double comp00,comp01,comp02,comp03;
         double comp10,comp11,comp12,comp13;
         double comp20,comp21,comp22,comp23;
         double comp30,comp31,comp32,comp33;
         string name;

         fourTensor(string forName)
         {
                comp00 = 0; comp01 = 0; comp02 = 0; comp03 = 0;
                comp10 = 0; comp11 = 0; comp12 = 0; comp13 = 0;
                comp20 = 0; comp21 = 0; comp22 = 0; comp23 = 0;
                comp30 = 0; comp31 = 0; comp32 = 0; comp33 = 0;
                name = forName;
         }
         void set(double forComp00, double forComp01, double forComp02, double forComp03,
                  double forComp10, double forComp11, double forComp12, double forComp13,
                  double forComp20, double forComp21, double forComp22, double forComp23,
                  double forComp30, double forComp31, double forComp32, double forComp33)
         {
            comp00 = forComp00; comp01 = forComp01; comp02 = forComp02; comp03 = forComp03;
            comp10 = forComp10; comp11 = forComp11; comp12 = forComp12; comp13 = forComp13;
            comp20 = forComp20; comp21 = forComp21; comp22 = forComp22; comp23 = forComp23;
            comp30 = forComp30; comp31 = forComp31; comp32 = forComp32; comp33 = forComp33;
         }

         void print(){
            std::cout << "THINSPIN::fourTensor " << name << "\n";
	    //            std::cout << "typeid(this).name()  " << typeid(this).name() << "\n";
         }
   };
}

#endif

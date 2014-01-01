#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<iostream>
#include<cfloat>
#include"../clr"
#include"../../../codes/UAL/src/UAL/Common/Def.hh"

int main(int argc,char*argv[]){

 if(argc!=3){
  printf("Usage: ./kappaD gD(esign) rD(esign)\n");
  printf("e.g.   ./kappaD +1.248107349 +40.0\n");
  exit(1);
 }

 long double oE9 = (long double) 1.e+9;
 long double c=UAL::clight;

 char*gDc;
 long double gD;
 gD = strtold(argv[1],&gDc);

 char*rDc;
 long double rD;
 rD = strtold(argv[2],&rDc);

 long double kappaD=1.L/gD;
 long double a=sqrt(1.L-kappaD*kappaD);   //   1.L-kappaD*kappaD;   //   a=sqrt(1.L-kappaD*kappaD);
 long double LcOver_k=1.L/a;
 long double mpG=UAL::pmass;
 long double ep=UAL::elemCharge;
 long double mp=mpG*ep*oE9/c/c;
 long double Ep=mp*c*c;
 long double Efac=1.L/gD;
 long double b=1.L/Efac/Efac-1.L;

 printf("\n" "\033[1m\033[31m" "DESIGN ORBIT\n");
 printf("gamma design/frozen  %+20.15Le\n",gD);
 printf("kappa design/frozen  %+20.15Le\n",kappaD);
 printf("radius design/frozen %+20.15Le\n",rD);
 printf("\n");
 printf("LcOver_k             %+20.15Le\n",LcOver_k);
 printf("Efac                 %+20.15Le\n",Efac);
 printf("\n");

 long double eps=0.L;
 if(1.L-kappaD*kappaD/Efac/Efac>0){
  eps=LcOver_k*sqrt(1.L-kappaD*kappaD/Efac/Efac);
 }
 else{
//printf("\"d\" 1.L-kappaD*kappaD/Efac/Efac %+20.15Le\n",1.L-kappaD*kappaD/Efac/Efac);
 }

 long double eps2=0.L;
 if(a*a-b+b*a*a > 0){
  eps2=1.L/a*sqrt(a*a-b+b*a*a);;
 }
 else{
//printf("\"d\" a*a-b+b*a*a %+20.15Le\n",a*a-b+b*a*a);
 }

 printf("a                    %+20.15Le\n",a);
 printf("b                    %+20.15Le\n",b);
 printf("eps                  %+20.15Le (\"d\"=1.L-kappaD*kappaD/Efac/Efac %+20.15Le) \n",eps,1.L-kappaD*kappaD/Efac/Efac);
 printf("eps2                 %+20.15Le (\"d\"=a*a-b+b*a*a %+20.15Le) \n",eps2,a*a-b+b*a*a);
 printf("\n");

 long double k=+2.6872238219e-09;
 long double L=k/c/a;
 printf("L                    %+20.15Le\n",L);

 long double thD0=L/gD/mp/rD/rD;
 printf("thD0                 %+20.15Le\n",thD0);
 printf(RESET);
}

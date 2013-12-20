#include "headers"

using namespace std;

int main(int argc,char * argv[]){

#include "inputs"
#include "parse"

 double b_0Hat_x=0.,b_0Hat_y=0.,b_0Hat_z=1.;
 double s_0Hat_x=0.,s_0Hat_y=0.,s_0Hat_z=1.;

 double   zHat_x=0.,  zHat_y=0.,  zHat_z=1.;

 double bHat_x=sin(theta_b)*cos(phi_b),bHat_y=sin(theta_b)*sin(phi_b),bHat_z=cos(theta_b);
 double bHat_norm=bHat_x*bHat_x+bHat_y*bHat_y+bHat_z*bHat_z;
 cerr << "bHat_norm (this should be 1): " << bHat_norm << "\n";

 //double inferredCosTheta_b=bHat_x*b_0Hat_x+b_0Hat_y*bHat_y+b_0Hat_z*bHat_z;
   double inferredCosTheta_b=bHat_x*zHat_x+bHat_y*zHat_y+bHat_z*zHat_z;
 cerr << "inferredCosTheta_b " << inferredCosTheta_b << "\n";

 double inferredTheta_b=acos(inferredCosTheta_b);
 cerr << "inferredTheta_b " << inferredTheta_b << "\n";

 double inferredPhi_b=atan2(bHat_y,bHat_x);
 cerr << "inferredPhi_b " << inferredPhi_b << "\n";

 double s_x=mag_s*sin(theta_s)*cos(phi_s),s_y=mag_s*sin(theta_s)*sin(phi_s),s_z=mag_s*cos(theta_s);
 double s_norm=s_x*s_x+s_y*s_y+s_z*s_z;
 cerr << "s_norm (this should be 1): " << s_norm << "\n";

 double inferredCosTheta_s=s_x*zHat_x+s_y*zHat_y+s_z*zHat_z;
 cerr << "inferredCosTheta_s " << inferredCosTheta_s << "\n";

 double inferredTheta_s=acos(inferredCosTheta_s);
 cerr << "inferredTheta_s " << inferredTheta_s << "\n";

 double inferredPhi_s=atan2(s_y,s_x);
 cerr << "inferredPhi_s " << inferredPhi_s << "\n";

 double gammaProbe=pow(1-betaProbe*betaProbe,-0.5);
 cerr << "gammaProbe " << gammaProbe << "\n";
// gammaProbe=1./sqrt(1-betaProbe*betaProbe);
// cerr << "gammaProbe " << gammaProbe << "\n";

 cerr << "================================================\n";

 double delGamma=gammaProbe-gamma_0;
 cerr << "delGamma          " << delGamma << "\n";

 double delGammaMaxF=E_0*gap/mcc_0/1.E9/4.;
 cerr << "delGammaMaxF      " << delGammaMaxF << "\n";
 cerr << "________________________________________________\n";

 cerr << "delThtaTilda      " << delThtaTilda << "\n";
 double delThtaTildaMaxF=gap/r_0/2.;                                          // 3.75E-4;
 cerr << "delThtaTildaMaxF  " << delThtaTildaMaxF << "\n";
 cerr << "________________________________________________\n";

 double delAlphaFactr=g/2.-1.+g/2./gamma_0/gamma_0;
// cerr << "delAlphaFactr " << delAlphaFactr << "\n";                           // 3.586? page 70

 double delAlphaTilda=delAlphaFactr*delGamma*delThtaTilda;                    // (284) page 70
 cerr << "delAlphaTilda     " << delAlphaTilda << "\n";

 double delAlphaTildaMaxF=delAlphaFactr*delGammaMaxF*delThtaTildaMaxF;        // (284) page 70
// tilda means quantity calculated in "instantaneous bend plane"
 cerr << "delAlphaTildaMaxF " << delAlphaTildaMaxF << "\n";
 cerr << "________________________________________________\n";

 double phi=-atan(dy_0/dx_0);                                                 // (286)
// cerr << "phi   " << phi   << " cos(phi)   " << cos(phi)   << " sin(phi)   " << sin(phi)  << "\n";
 double rho_0=pow((dx_0*dx_0+dy_0*dy_0),0.5);                                 // page 71
// cerr << "rho_0 " << rho_0 << " dx_0/rho_0 " << dx_0/rho_0 << " dy_0/rho_0 " << dy_0/rho_0 << "\n";

 matrix L,R;

 cerr << "================================================\n";

 L.setRoll(+phi);
 L.show("left roll (L:  do coordinate transformation) ");
 cerr << "================================================\n";

 R.setRoll(-phi);
 R.show("right roll (R: undo coordinate transformation) ");
 cerr << "================================================\n";

 matrix P;
 P.setYaw(delAlphaTilda);
 P.show("pure spin precession yaw (P)");
 cerr << "================================================\n";

 matrix T;
 T=L*P*R;
 T.show("overall spin transformation (T = L*P*R)");
 cerr << "================================================\n";

 matrix Tm1=T.deltaFromId();
 Tm1.show("overall spin transformation minus 1 (Tm1)");
 cerr << "================================================\n";

 matrix Pm1=P.deltaFromId();
 matrix LPm1R=L*Pm1*R;
 LPm1R.show("LPm1R");
 cerr << "================================================\n";

 return (int)0;
}

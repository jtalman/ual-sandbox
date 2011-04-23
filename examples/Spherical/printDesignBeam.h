  std::cout << setiosflags( ios::showpos   );
  std::cout << setiosflags( ios::uppercase );
  std::cout << setiosflags( ios::scientific );
//std::cout << setw( 11 );
  std::cout << setfill( ' ' );
  std::cout << setiosflags( ios::left );
  std::cout << setprecision(13) ;

  std::cout << "e0=cba.getEnergy()      " << e0   << "\n";
  std::cout << "m0=cba.getMass()        " << m0   << "\n";
  std::cout << "q0=cba.getCharge()      " << q0   << "\n";
  std::cout << "t0=cba.getElapsedTime() " << t0   << "\n";
  std::cout << "f0=cba.getRevfreq()     " << f0   << "\n";
  std::cout << "M0=cba.getMacrosize()   " << M0   << "\n";
  std::cout << "G0=cba.getG()           " << G0   << "\n";

  std::cout << "Eel0                    " << Eel0 << "\n";
  std::cout << "Rsxf                    " << Rsxf << "\n";

  std::cout << "GeVperJ                 " << GeVperJ   << "\n";
  std::cout << "Eel0MKS                 " << Eel0MKS   << "\n";

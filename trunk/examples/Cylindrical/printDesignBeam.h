  std::cout << setiosflags( ios::showpos   );
  std::cout << setiosflags( ios::uppercase );
  std::cout << setiosflags( ios::scientific );
//std::cout << setw( 11 );
  std::cout << setfill( ' ' );
  std::cout << setiosflags( ios::left );
  std::cout << setprecision(7) ;

  std::cout << "e0=cba.getEnergy()      " << e0 << "\n";
  std::cout << "m0=cba.getMass()        " << m0 << "\n";
  std::cout << "q0=cba.getCharge()      " << q0 << "\n";
  std::cout << "t0=cba.getElapsedTime() " << t0 << "\n";
  std::cout << "f0=cba.getRevfreq()     " << f0 << "\n";
  std::cout << "M0=cba.getMacrosize()   " << M0 << "\n";
  std::cout << "G0=cba.getG()           " << G0 << "\n";

  std::cout << "L0                      " << L0 << "\n";
  std::cout << "E0                      " << E0 << "\n";
  std::cout << "R0                      " << R0 << "\n";

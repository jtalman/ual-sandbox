std::cout << "#################################   Set Beam Attributes\n";
  shell.setBeamAttributes(UAL::Args() << UAL::Arg("mass",                 m0));
  shell.setBeamAttributes(UAL::Args() << UAL::Arg("energy",               e0));
  shell.setBeamAttributes(UAL::Args() << UAL::Arg("charge",               q0));
  shell.setBeamAttributes(UAL::Args() << UAL::Arg("elapsedTime",          t0));
  shell.setBeamAttributes(UAL::Args() << UAL::Arg("frequency",            f0));
  shell.setBeamAttributes(UAL::Args() << UAL::Arg("macrosize",            M0));

  shell.setBeamAttributes(UAL::Args() << UAL::Arg("gyromagnetic",         G0));
  shell.setBeamAttributes(UAL::Args() << UAL::Arg("gFactor",              g0));

  shell.setBeamAttributes(UAL::Args() << UAL::Arg("designAngularMomentum",L0));
  shell.setBeamAttributes(UAL::Args() << UAL::Arg("designElectricField",  El0));
  shell.setBeamAttributes(UAL::Args() << UAL::Arg("designRadius",         IA));
std::cout << "#################################   Set Beam Attributes\n";

 double mass   = 0.93827231; // proton rest mass
 double m0=mass;
 double chge   = UAL::elemCharge;       // 1.6e-19   ; // proton charge

 double gamma0 = atof(argv[3]);
 double v0= UAL::clight*sqrt(1-1/gamma0/gamma0);
 double energy = gamma0*m0;
 double e0=energy;
 double p0 = gamma0*m0*v0;
 double R0=atof(argv[4]);
 std::string sxfFile = "./data/";
 sxfFile += argv[2];
 sxfFile += ".sxf";
 std::string outputFile = "./out/cpp/";
 outputFile += argv[2];
 outputFile += ".sxf";
 std::string mapFile = "./out/cpp/";
 mapFile += argv[2];
 mapFile += ".map1";
 std::string twissFile = "./out/cpp/";
 twissFile += argv[2];
 twissFile += ".twiss";
 std::string apdfFile = argv[1];
 double probe__dx0 = atof(argv[6]);
 double probe_dpx0 = atof(argv[7]);
 double probe__dy0 = atof(argv[8]);
 double probe_dpy0 = atof(argv[9]);
 double probe_cdt0 = atof(argv[10]);

 double probe__dz0 = probe_cdt0;
 double probe_dpz0 = 0;

 double probe_X0   = R0 + probe__dx0;
 double probe_Y0   = probe__dy0;
 double probe_Z0   = probe__dz0;

 double probePX0   = probe_dpx0;
 double probePY0   = probe_dpy0;
 double probePZ0   = p0 + probe_dpz0;

 double probe_r0   = sqrt(probe_X0*probe_X0+probe_Y0*probe_Y0+probe_Z0*probe_Z0);
 double probeP0    = sqrt(probePX0*probePX0+probePY0*probePY0+probePZ0*probePZ0);
 std::string orbitFile = "./out/cpp/";
 orbitFile += argv[2];
 orbitFile += ".orbit";

 int split = atoi(argv[11]);
 double probeEscr0 = sqrt(probeP0*probeP0+m0*m0)+chge*R0*log(probe_r0/R0) - sqrt(p0*p0+m0*m0);

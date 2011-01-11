   std::string sxfFile = argv[1];
// std::string sxfFile = "./data/";
// sxfFile += argv[2];
// sxfFile += ".sxf";

 std::string outputFile = "./out/cpp/";
 outputFile += mysxfbase;
 outputFile += ".sxf";
 std::string mapFile = "./out/cpp/";
 mapFile += mysxfbase;
 mapFile += ".map1";
 std::string twissFile = "./out/cpp/";
 twissFile += mysxfbase;
 twissFile += ".twiss";
 std::string apdfFile = "./data/eteapot.apdf";
 double probe__dx0 = .1;
 double probe_dpx0 = 1e-3;
 double probe__dy0 = .1;
 double probe_dpy0 = 1e-4;
 double probe_cdt0 = 1e-3;

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
 orbitFile += mysxfbase;
 orbitFile += ".orbit";

 int split = 0;
 double probeEscr0 = sqrt(probeP0*probeP0+m0*m0)+q0*R0*log(probe_r0/R0) - sqrt(p0*p0+m0*m0);


int run(){

  gSystem->Load("./lib/linux/libBTFShell.so");


  BTFShell shell;

  // ************************************************************************
  cout << "1. Reads the SXF file." << endl;
  // ************************************************************************

  shell.readSXF("./data/rhic_injection.sxf", "./out");

  // ************************************************************************
  cout << "2. Defines a lattice." << endl;
  // ************************************************************************

  shell.use("blue");

 // ************************************************************************
  cout << "3. Makes survey and gets a lattice length . " << endl;
  // ************************************************************************

  double suml = shell.getLength();
  std::cout << "lattice length: " << suml << " [m]" << std::endl; 

  // ************************************************************************
  cout << "4. Reads the APDF file." << endl;
  // ************************************************************************

  shell.readAPDF("./data/tibetan.apdf");

  // ************************************************************************
  cout << "5. Defines beam attributes (energy, mass, charge)." << endl;
  // ************************************************************************

  PAC::Bunch bunch;

  double e = 10.0;
  double m = bunch.getMass();
  double v0byc = sqrt(e*e - m*m)/e;

  double revFreq =  v0byc*UAL::clight/suml;
  std::cout << "revolution frequency: " << revFreq << " [1/s]" << std::endl;

  bunch.setEnergy(e);
  bunch.setRevfreq(revFreq);

  // ************************************************************************
  cout<<"6. Initializing input particle distribution"<<endl;  
  // ************************************************************************

  int    nparts      = 10000;
  double ctSigma     = 0.25;
  double deSigma     = 2.0e-4;
  int    iseed       = -100;

  bunch.resize(nparts);
  shell.generateBunch(bunch, 1.4*2.0*ctSigma, 1.4*2.0*deSigma, iseed);

  TNtupleD inputBunch("inputBunch",
		      "Input particle distribution to tracker","x:x':y:y':ct:dp:flag");
  UAL::bunch2Ntuple(bunch, inputBunch);
 
  // ************************************************************************
  cout << "7. Defines Kicker and Bpm attributes . " << endl;
  // ************************************************************************

  double kick     = 1.0e-5;
  int nFreq       = 245e+6/revFreq;
  double fracFreq = 0.2;
  double phase    = 0.0;

  AIM::BTFKicker kicker;
  kicker.setHKick(kick, nFreq, fracFreq, phase);

  double freqLo = (0.180 + nFreq) *revFreq;
  double freqHi = (0.230 + nFreq) *revFreq;;
  int    nfreqs = 100;  

  double ctBin  = suml/nFreq/40;

  AIM::BTFBpm bpm;
  bpm.setHFreqRange(freqLo, freqHi, nfreqs);
  bpm.setCtBin(ctBin);

  // ************************************************************************
  cout << "8. BTF tracking. " << endl;
  // ************************************************************************

  int nturns = 200;

  UAL::AcceleratorPropagator* accTracker = shell.getTracker();

  AIM::BTFBpmCollector& bpmCollector = AIM::BTFBpmCollector::getInstance();
  bpmCollector.clear();

  for(int iturn = 0; iturn < nturns; iturn++){
    kicker.setTurn(iturn);
    kicker.propagate(bunch);
    accTracker->propagate(bunch);
    bpm.propagate(bunch);
  }


  // ************************************************************************
  cout << "9. Copies UAL output to ROOT containers . " << endl;
  // ************************************************************************

  // Output bunch
  TNtupleD outputBunch("outputBunch",
		       "Final particle distribution","x:x':y:y':ct:dp:flag");
  UAL::bunch2Ntuple(bunch, outputBunch);


  // Line density
  TH2D* densityTH2D = new TH2D("densityTH2D", 
			       "Line density", 
			       nturns, 0.0, nturns, 
			       10*(ctSigma/ctBin), -5*ctSigma, 5*ctSigma );
  shell.getLineDensity(*densityTH2D); 

  // Horizontal dipole terms
  TH2D* hDipoleTH2D = new TH2D("hDipoleTH2D", "X Dipole Term", 
			       nturns, 0.0, nturns, 
			       10*(ctSigma/ctBin), -5*ctSigma, 5*ctSigma);
  shell.getHDipoleTerm(*hDipoleTH2D); 

  // Spectrum
  TH2D* hSpecTH2D = new TH2D("hSpecTH2D", "H Spectrum", nturns, 0.0, nturns, 
			     nfreqs, freqLo/revFreq - nFreq, freqHi/revFreq - nFreq);
  shell.getHSpectrum(*hSpecTH2D, revFreq, nFreq); 
 
  // ************************************************************************
  cout << "10. Draw output." << endl;
  // ************************************************************************

  // input bunch
  TCanvas* c1 = new TCanvas("c1","Input Bunch");
  c1->Divide(2,2);
  c1->cd(1);
  inputBunch.Draw("ct:dp", "", "lego");
  c1->cd(2);
  inputBunch.Draw("ct:dp");
  c1->cd(3);
  inputBunch.Draw("ct");
  c1->cd(4);
  inputBunch.Draw("dp");
 
  // line density
  TCanvas* c2 = new TCanvas("c2","Line Density");
  c2->cd(1);
  densityTH2D->SetFillColor(17);
  densityTH2D->Draw("lego2");

  // dipole terms
  TCanvas* c3 = new TCanvas("c3","Dipole Term");
  c3->cd(1);
  hDipoleTH2D->SetFillColor(17);
  hDipoleTH2D->Draw("surf1"); 

  // spectrum
  TCanvas* c5 = new TCanvas("c5","Spectrum");
  c5->cd(1);
  hSpecTH2D->SetFillColor(17);
  hSpecTH2D->Draw("lego2"); 

  // output bunch
  TCanvas* c4 = new TCanvas("c4","Output Bunch");
  c4->Divide(2,2);
  c4->cd(1);
  outputBunch.Draw("ct:dp", "", "lego");
  c4->cd(2);
  outputBunch.Draw("x*1e3:x'*1e3");
  c4->cd(3);
  outputBunch.Draw("y*1e3:y'*1e3");
  c4->cd(4);
  outputBunch.Draw("dp");

}

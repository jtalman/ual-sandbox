#include <string>
/* ****************************************************************************
   *                                                                          *
   *    This is asample script that demonstrates a use of a simple shell.     *
   *    This will be expanded at some point to use collimators but            *
   *    for now stands as an example of the use of ROOT and the UAL.          *
   *                                                                          *
   *    This script reading in an SXF file, associates algorithms with        *
   *    an APDF file, makes a bunch inside of a ROOT TNtuple. Then it         *
   *    tracks that bunch around the ring.  Finally it plots the input and    *
   *    output phase spaces                                                   *
   *                                                                          *
   *    Author: Ray Fliller III and Nikolay Malitsky                          *
   *                                                                          *
   *                                                                          *
   *                                                                          *
   ****************************************************************************
*/


int run(){  
  
  UAL::RootShell shell;

  
  // ************************************************************************
  cout << "1. Reads the SXF file." << endl;
  // ************************************************************************
  
  shell.readSXF("./data/rhic_injection.sxf", "./out");
  
  // ************************************************************************
  cout << "2. Defines a lattice." << endl;
  // ************************************************************************
  
  shell.use("blue");
    
  // ************************************************************************
  cout << "3. Defines beam attributes." << endl;
  // ************************************************************************
  
  PAC::BeamAttributes ba;

  ba.setEnergy(250.0);
  ba.setMass(0.9382796);
  ba.setCharge(1.0);

  shell.setBeamAttributes(ba);
  
  // ************************************************************************
  cout << "4. Reads the APDF file." << endl;
  // ************************************************************************
  
  shell.readAPDF("./data/tracker.apdf");
  
  // ************************************************************************
  cout<<"5. Initializing input particle distribution"<<endl;  
  // ************************************************************************

  PAC::Bunch bunch(1000);
  
  shell.generateBunch(bunch); 

  TNtupleD input("input","Input particle distribution","x:x':y:y':ct:dp:flag");
  UAL::bunch2Ntuple(bunch, input);
  
  // ************************************************************************
  cout << "6. Track particles." << endl;
  // ************************************************************************
  
  shell.track(bunch);
  
  // ************************************************************************
  cout << "7. Analyze output." << endl;
  // ************************************************************************

  TNtupleD output("output","Output of Tracked Particles","x:x':y:y':ct:dp:flag");
  UAL::bunch2Ntuple(bunch, output); 
  

  //draw plots of the input distribution.
  TCanvas* c1 = new TCanvas("c1","Input Phase Spaces");  //make the Canvas
  c1->Divide(2,2);                                       //divide into 4
  c1->cd(1);                                             //go to the first division
  input.Draw("x'*1e6:x*1e3");                           // draw the horizontal phase space in mm and urad
  h=(TH1*)(gPad->FindObject("htemp"));                   //get a pointer to the histogram.
  h->SetXTitle("x (mm)");                                //the next lines make it pretty
  h->SetYTitle("x' (#murad)");
  h->GetXaxis()->CenterTitle();
  h->GetYaxis()->CenterTitle();
  h->SetTitle("Horizontal Phase Space");

  c1->cd(2);                                            //go the to the second and repeat for y
  input.Draw("y'*1e6:y*1e3");
  h=(TH1*)(gPad->FindObject("htemp"));
  h->SetXTitle("y (mm)");
  h->SetYTitle("y' (#murad)");
  h->GetXaxis()->CenterTitle();
  h->GetYaxis()->CenterTitle();
  h->SetTitle("Vertical Phase Space");

  c1->cd(3);
  input.Draw("ct");
  h=(TH1*)(gPad->FindObject("htemp"));
  h->SetXTitle("ct (m)");
  h->GetXaxis()->CenterTitle();
  h->SetTitle("Longitudinal Position");

  c1->cd(4);
  input.Draw("dp");  
  h=(TH1*)(gPad->FindObject("htemp"));
  h->SetXTitle("#delta=#frac{#Delta p}{p_{0}}");
  h->GetXaxis()->CenterTitle();
  h->SetTitle("Momentum Deviation");


 //draw plots of the output distribution. 
  TCanvas* c2 = new TCanvas("c2","Output Phase Spaces");
  c2->Divide(2,2);
  c2->cd(1);
  output.Draw("x'*1e6:x*1e3");
  h=(TH1*)(gPad->FindObject("htemp"));
  h->SetXTitle("x (mm)");
  h->SetYTitle("x' (#murad)");
  h->GetXaxis()->CenterTitle();
  h->GetYaxis()->CenterTitle();
  h->SetTitle("Horizontal Phase Space");

  c2->cd(2);
  output.Draw("y'*1e6:y*1e3");
  h=(TH1*)(gPad->FindObject("htemp"));
  h->SetXTitle("y (mm)");
  h->SetYTitle("y' (#murad)");
  h->GetXaxis()->CenterTitle();
  h->GetYaxis()->CenterTitle();
  h->SetTitle("Vertical Phase Space");

  c2->cd(3);

  output.Draw("ct");
  h=(TH1*)(gPad->FindObject("htemp"));
  h->SetXTitle("ct (m)");
  h->GetXaxis()->CenterTitle();
  h->SetTitle("Longitudinal Position");

  c2->cd(4);
  output.Draw("dp");
  h=(TH1*)(gPad->FindObject("htemp"));
  h->SetXTitle("#delta=#frac{#Delta p}{p_{0}}");
  h->GetXaxis()->CenterTitle();
  h->SetTitle("Momentum Deviation");

 // ************************************************************************
  cout << "8. Write to Disk." << endl;
  // ************************************************************************


  //make Postscipt files of the plots
  c1->SaveAs("c1.ps");
  c2->SaveAs("c2.ps");


  //save to a root file.
  TFile *file = new TFile("demo.root","NEW");  //make a new file and open it.
  input.Write();                              //write the input.
  output.Write();                             //write the output
  c1->Write();                                 //write the input plots.
  c2->Write();                                 //write the output plots.
  file->ls();                                  //print the file contents.
  file->Close();                               //close it up

  return 1;

  
}

{
/*  This example shows how to use the CORAANT TTurnPlot class to read fort.8 files
    as specified on page 39 of the fortran TEAPOT manual
    To use this script, inside of root do:

    gROOT->Reset();
    .X PlotCESR.cc

*/


  gSystem->Load("$UAL/tools/lib/linux/libCORAANT.so");  //load the library
  using namespace CORAANT;                              

  file = new TFile("CESR.root","NEW","",9);
  track=new TTurnPlot("track","Tracking results from CESR","cesr.fort.8");  //reads in the cesr.fort.8 file
  track->Write();  //writes to the root file

  c=new TCanvas("c","Phase space plots");
  c->Divide(2,1);
  c->cd(1);
  track->Draw("x':x");
  c->cd(2);
  track->Draw(Form("(%lf*x+sqrt(%lf)*x'):x/sqrt(%lf)",track->GetAlphaX()/sqrt(track->GetBetaX()),track->GetBetaX(),track->GetBetaX()));
 // Form returns a character string as specified by the printf format.
 //the above line draws the normalized phase space
  


  c2=new TCanvas("c2","Turn By Turn Data");
  c2->Divide(2,1);
  c2->cd(1);
  track->MultiGraph("x,y:Turn","N==5");
  c2->cd(2);
  track->Graph("x:Turn","N==2");

  c3=new TCanvas("c3","Tune Plots");
  c3->Divide(2,2);
  c3->cd(1);
  cout<<"making graph 1"<<endl;
  track->TuneGraph("x","N==0");
  c3->cd(2);
  cout<<"making graph 2"<<endl;
  track->TuneGraph("x","N==1");
  c3->cd(3);
  cout<<"making graph 3"<<endl;
  track->TuneGraph("x","N==4");
  c3->cd(4);
  cout<<"making graph 4"<<endl;
  track->TuneGraph("x","N==5");

  c4=new TCanvas("c4","Tune Space");
  tunes = new TTuneSpace("Tunes","Tunes of Tracked particles");
  Float_t xmax,ymax;
  TTuneGraph *g;
// go throught each particle and get the peak tune in each tune spectrum for each plane
  for(int i=0;i<track->GetParticles();i++){
    xmax=ymax=0; 
    track->TuneGraph("x",Form("N==%i",i));  //get the x plane and its tune.
    g=(TTuneGraph *)(gPad->FindObject(Form("Particle:%i",i)));
    xmax=g->GetMaxPeakPositionX();
    gPad->Clear();
    track->TuneGraph("y",Form("N==%i",i)); //get the y plane and its tune.
    g=(TTuneGraph *)(gPad->FindObject(Form("Particle:%i",i)));
    ymax=g->GetMaxPeakPositionX(); //the X here is the X axis of g!!!
    tunes->Fill(xmax,ymax);
    gPad->Clear();
  }

  gPad->SetLogy(0); //reset linear scale on this canvas
  tunes->SetResonancesOrder(4);
  tunes->Draw();

  c->Write();
  c2->Write();
  c3->Write();
  c4->Write();
  file->ls();
 
}

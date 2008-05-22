#include <qapplication.h>

#include "AIM/BPM/MonitorCollector.hh"
#include "UAL/ROOT/Viewers/MonitorFFTViewer.hh"


UAL::ROOT::MonitorFFTViewer::MonitorFFTViewer(UAL::QT::BasicPlayer* player)
  : UAL::ROOT::BasicViewer()
{

  p_player   = player;
  m_bpmIndex = 0;

  viewer->GetCanvas()->Divide(0, 2);
  viewer->GetCanvas()->SetLogy();

  xTunes = new TGraph(); 
  xTunes->SetTitle("Horizontal spectrum");
  xTunes->SetBit(kCanDelete);
  xTunes->SetMarkerStyle(2);
  xTunes->SetLineColor(31);

  yTunes = new TGraph(); 
  yTunes->SetTitle("Vertical spectrum");
  yTunes->SetBit(kCanDelete);
  yTunes->SetMarkerStyle(20);
  yTunes->SetLineColor(31);

  updatePoints();

  viewer->cd(1);
  gPad->SetLogy();
  xTunes->GetXaxis()->SetLimits(0.0, 0.5); 
  gPad->SetLogy();
  xTunes->Draw("AL");
  gPad->SetLogy();

  viewer->cd(2);
  gPad->SetLogy();
  yTunes->GetXaxis()->SetLimits(0.0, 0.5);
  gPad->SetLogy();
  yTunes->Draw("AL"); 
  gPad->SetLogy(); 

  viewer->GetCanvas()->cd();
  viewer->GetCanvas()->Modified();
  viewer->GetCanvas()->Update();
}

void UAL::ROOT::MonitorFFTViewer::closeEvent(QCloseEvent* ce)
{
  ce->accept();
  p_player->removeViewer("UAL::ROOT::MonitorFFTViewer");
}

void UAL::ROOT::MonitorFFTViewer::updatePoints()
{

  AIM::MonitorCollector& mc =  AIM::MonitorCollector::getInstance();

 std::map<int, AIM::Monitor*>::iterator ibpms = mc.getAllData().begin();  

  if(ibpms == mc.getAllData().end()) return;

  std::vector<double> freq, hspec, vspec;
  mc.fft(ibpms->second->getIndex(), freq, hspec, vspec);

  xTunes->Set(hspec.size());
  for(unsigned int i = 0; i < hspec.size(); i++){
    xTunes->SetPoint(i, freq[i], hspec[i]);
  }

  yTunes->Set(vspec.size());
  for(unsigned int i = 0; i < vspec.size(); i++){
    yTunes->SetPoint(i, freq[i], vspec[i]);
  }

}

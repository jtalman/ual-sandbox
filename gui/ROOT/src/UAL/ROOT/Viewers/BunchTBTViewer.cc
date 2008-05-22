#include <qapplication.h>

#include "AIM/BPM/Monitor.hh"
#include "AIM/BPM/MonitorCollector.hh"

#include "UAL/ROOT/Viewers/BunchTBTViewer.hh"

UAL::ROOT::BunchTBTViewer::BunchTBTViewer(UAL::QT::BasicPlayer* player)
  : UAL::ROOT::BasicViewer()
{

  p_player   = player;

  m_bpmIndex = 0;
 
  viewer->GetCanvas()->Divide(0, 3);
  int turns = p_player->getTurns();

  xTBT = new TGraph(turns); 
  xTBT->SetTitle("Horizontal");
  xTBT->SetBit(kCanDelete);
  xTBT->SetMarkerStyle(2);
  xTBT->SetLineColor(31);

  yTBT = new TGraph(turns); 
  yTBT->SetTitle("Vertical");
  yTBT->SetBit(kCanDelete);
  yTBT->SetMarkerStyle(2);
  yTBT->SetLineColor(31);

  ctTBT = new TGraph(turns); 
  ctTBT->SetTitle("Longitudinal");
  ctTBT->SetBit(kCanDelete);
  ctTBT->SetMarkerStyle(2);
  ctTBT->SetLineColor(31);

  for(int it = 0; it < turns; it++){
    xTBT->SetPoint(it, it, 0.0);
    yTBT->SetPoint(it, it, 0.0);
    ctTBT->SetPoint(it, it, 0.0);
  }

  viewer->cd(1);

  xTBT->Draw("AL");
  xTBT->GetXaxis()->SetRange(0, 0);
  xTBT->GetXaxis()->SetLimits(0, turns); 

  viewer->cd(2);

  yTBT->Draw("AL");
  yTBT->GetXaxis()->SetRange(0, 0);
  yTBT->GetXaxis()->SetLimits(0, turns); 

  viewer->cd(3);

  ctTBT->Draw("AL");
  ctTBT->GetXaxis()->SetRange(0, 0);
  ctTBT->GetXaxis()->SetLimits(0, turns); 


  int turn = getTurn();
  if(turn > 0) updateViewer(turn);

  viewer->GetCanvas()->cd();

  viewer->GetCanvas()->Modified();
  viewer->GetCanvas()->Update();
}

void UAL::ROOT::BunchTBTViewer::updateViewer(int turn)
{

  int minTurn = minTurn -1000 > 0 ? minTurn - 1000 : 0;
  int maxTurn = turn;
  

  updatePoints(turn);

  viewer->cd(1);
  xTBT->Draw("AL");
  xTBT->GetXaxis()->SetRange(minTurn, maxTurn);

  viewer->cd(2);
  yTBT->Draw("AL");
  yTBT->GetXaxis()->SetRange(minTurn, maxTurn);

  viewer->cd(3);
  ctTBT->Draw("AL");
  ctTBT->GetXaxis()->SetRange(minTurn, maxTurn);

  viewer->GetCanvas()->Modified();
  viewer->GetCanvas()->Update();

}

void UAL::ROOT::BunchTBTViewer::closeEvent(QCloseEvent* ce)
{
  ce->accept();
  p_player->removeViewer("UAL::ROOT::BunchTBTViewer");

}

void UAL::ROOT::BunchTBTViewer::updatePoints(int turn)
{

  std::cout << "TBTViewer updates points, turn is  " << turn << std::endl;

  std::map<int, AIM::Monitor*>::iterator ibpms = 
    AIM::MonitorCollector::getInstance().getAllData().begin(); // find(m_bpmIndex);

  if(ibpms == AIM::MonitorCollector::getInstance().getAllData().end()) 
    return;

  AIM::Monitor* bpm = ibpms->second;

  std::list<PAC::Position>::iterator it;

  int counter = 0;
  for(it = bpm->getData().begin(); it != bpm->getData().end(); it++){
    xTBT->SetPoint(counter, counter, it->getX());
    counter++;
  }

  counter = 0;
  for(it = bpm->getData().begin(); it != bpm->getData().end(); it++){
    yTBT->SetPoint(counter, counter, it->getY());
    counter++;
  }

  counter = 0;
  for(it = bpm->getData().begin(); it != bpm->getData().end(); it++){
    ctTBT->SetPoint(counter, counter, it->getCT());
    counter++;
  }
}

int UAL::ROOT::BunchTBTViewer::getTurn()
{
  std::map<int, AIM::Monitor*>::iterator ibpms = 
    AIM::MonitorCollector::getInstance().getAllData().begin(); 

  if(ibpms == AIM::MonitorCollector::getInstance().getAllData().end()) 
    return 0;

  return ibpms->second->getData().size() - 1;
}




#include <qapplication.h>

#include "UAL/Common/Def.hh"

#include "AIM/BPM/PoincareMonitor.hh"
#include "AIM/BPM/PoincareMonitorCollector.hh"

#include "UAL/QT/Player/TurnCounter.hh"
#include "UAL/ROOT/Viewers/PoincareCtDeViewer.hh"

UAL::ROOT::PoincareCtDeViewer::PoincareCtDeViewer(UAL::QT::BasicPlayer* player)
  : UAL::ROOT::PoincareViewer(player)
{

  /* defined in PoincareViewer
  p_player     = player;

  m_prevTurn   =  1000000000;

  m_bpmIndex   = 0;
  m_points     = p_player->getTurns();
  */

  // std::cout << "points = " << m_points << std::endl;

  m_tinyGraph = new TGraph(2); 
  m_tinyGraph->SetTitle("Poincare CT-DE Plot");
  m_tinyGraph->GetXaxis()->SetTitle("x, m");
  m_tinyGraph->GetYaxis()->SetTitle("px/p0, rad");
  m_tinyGraph->GetYaxis()->CenterTitle();
  m_tinyGraph->SetMarkerStyle(11);
  m_tinyGraph->SetMarkerColor(8);
  m_tinyGraph->SetBit(kCanDelete);

  double xmax = 1.e-9, pxmax = 1.e-9;

  m_tinyGraph->SetPoint(0,  xmax,  pxmax);
  m_tinyGraph->SetPoint(0, -xmax, -pxmax);

  UAL::QT::TurnCounter* tc = UAL::QT::TurnCounter::getInstance();
  int turn = tc->getTurn();

  if (turn > 0) updateViewer(turn);
  else {

    viewer->GetCanvas()->cd();

    m_tinyGraph->Draw("AP");
    m_tinyGraph->GetXaxis()->SetTitle("x, m");
    m_tinyGraph->GetYaxis()->SetTitle("px/p0, rad");
    m_tinyGraph->GetYaxis()->CenterTitle();
    m_tinyGraph->SetMarkerStyle(11);
    m_tinyGraph->SetMarkerColor(8);

    viewer->GetCanvas()->Modified();
    viewer->GetCanvas()->Update();
  }

}


void UAL::ROOT::PoincareCtDeViewer::updateViewer(int turn)
{

  // std::cout << "ROOT::PoincareCtDeViewer::updateViewer turn:" << turn  << std::endl;
  if(turn < 1) return;

  updatePoints(turn);

  viewer->cd();

  m_tinyGraph->Draw("AP"); 
  m_tinyGraph->GetXaxis()->SetTitle("ct, m");
  m_tinyGraph->GetYaxis()->SetTitle("de/p0");
  m_tinyGraph->GetYaxis()->CenterTitle();
  m_tinyGraph->SetMarkerStyle(11);
  m_tinyGraph->SetMarkerColor(8);

  for(unsigned int i=0; i < m_graphs.size(); i++){
    m_graphs[i]->Draw("P");
  }

  // std::cout << "ROOT::PoincareXPXViewer::updateViewer " << std::endl;

  viewer->GetCanvas()->Modified();
  viewer->GetCanvas()->Update();

}

void UAL::ROOT::PoincareCtDeViewer::closeEvent(QCloseEvent* ce)
{

  ce->accept();
  p_player->removeViewer("UAL::ROOT::PoincareCtDeViewer");

}

void UAL::ROOT::PoincareCtDeViewer::initPoints()
{

}

void UAL::ROOT::PoincareCtDeViewer::updatePoints(int turn)
{


  std::map<int, AIM::PoincareMonitor*>::iterator ibpms = 
    AIM::PoincareMonitorCollector::getInstance().getAllData().begin(); // find(m_bpmIndex);

  if(ibpms == AIM::PoincareMonitorCollector::getInstance().getAllData().end()) 
    return;

  AIM::PoincareMonitor* bpm = ibpms->second;

  std::list<PAC::Bunch>::iterator it;

  if(m_prevTurn > turn){
 
    for(unsigned int i=0; i < m_graphs.size(); i++){
      delete m_graphs[i];
    }

    m_graphs.clear();

    it = bpm->getData().begin();
    int ngraphs  = it->size();

    m_graphs.resize(ngraphs);

    for(unsigned int i=0; i < m_graphs.size(); i++){
      m_graphs[i] = new TGraph(m_points + 1); 
      m_graphs[i]->SetMarkerStyle(7);
      m_graphs[i]->SetMarkerColor(31+i);
      m_graphs[i]->SetBit(kCanDelete);

      for(int j=0; j < m_points; j++){
	m_graphs[i]->SetPoint(j, 0.0, 0.0);
      }

    } 

    for(unsigned int i=0; i < m_graphs.size(); i++){
	m_graphs[i]->Draw("P");
    }

    m_prevTurn = turn;

  }

  int counter = 0;
  double xmax = 0.0, pxmax = 0.0;

  for(it = bpm->getData().begin(); it != bpm->getData().end(); it++){
    PAC::Bunch& bunch = *it;

    for(int i = 0; i < bunch.size(); i++) {

      if(bunch[i].isLost()) {
	m_graphs[i]->SetPoint(counter, 0.0, 0.0);
	continue;
      }

      PAC::Position& p = bunch[i].getPosition();
      double x  = p.getCT();
      double px = p.getDE();
      if(fabs(x) > xmax)   xmax  = fabs(x);
      if(fabs(px) > pxmax) pxmax = fabs(px);
      m_graphs[i]->SetPoint(counter, x, px);


    }

    counter++;
  }

  // std::cout << "xmax = " << xmax << ", pxmax = " << pxmax << std::endl;

  m_tinyGraph->SetPoint(0, xmax, pxmax);
  m_tinyGraph->SetPoint(1, -xmax, -pxmax);

}


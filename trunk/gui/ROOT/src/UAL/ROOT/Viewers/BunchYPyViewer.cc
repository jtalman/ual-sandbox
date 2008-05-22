#include <qapplication.h>

#include "UAL/Common/Def.hh"
#include "UAL/ROOT/Viewers/BunchYPyViewer.hh"


UAL::ROOT::BunchYPyViewer::BunchYPyViewer(UAL::QT::BasicPlayer* player, 
					  PAC::Bunch* bunch)
  : UAL::ROOT::BasicViewer()
{

  p_player     = player;
  p_bunch      = bunch;


  findLimits();

  m_ybins  = 100;
  m_pybins = 100;
  ypy = new TH2F("ypy", "Y-PY Phase Space", 
		 m_ybins,  -1.5*m_yMax,  +1.5*m_yMax, 
		 m_pybins, -1.5*m_pyMax, +1.5*m_pyMax);

  ypy->Draw();
  ypy->SetBit(kCanDelete);

  ypy->SetOption("colz");
  ypy->GetXaxis()->SetTitle("y, m");
  ypy->GetYaxis()->SetTitle("py/p0");
  ypy->GetYaxis()->CenterTitle();

  updatePoints();

  viewer->GetCanvas()->cd();

  viewer->GetCanvas()->Modified();
  viewer->GetCanvas()->Update();
}


void UAL::ROOT::BunchYPyViewer::updateViewer(int turn)
{

  updatePoints();

  viewer->cd();

  viewer->GetCanvas()->Modified();
  viewer->GetCanvas()->Update();

}

void UAL::ROOT::BunchYPyViewer::closeEvent(QCloseEvent* ce)
{

  ce->accept();
  p_player->removeViewer("UAL::ROOT::BunchYPyViewer");

}

void UAL::ROOT::BunchYPyViewer::updatePoints()
{
  if(!p_bunch) return;

  int size = p_bunch->size();

  ypy->Reset("");

  for(int i=0; i < size; i++) {

    if((*p_bunch)[i].isLost()) {
      continue;
    }

    PAC::Position& p = (*p_bunch)[i].getPosition();
    ypy->Fill(p.getY(), p.getPY());
  } 

  // std::cout << "ctde viewer: update points " << std::endl;

}

void UAL::ROOT::BunchYPyViewer::findLimits()
{
  m_yMax  = +1.e-20;
  m_pyMax = +1.e-20;

  if(!p_bunch) return;

  int size = p_bunch->size();

  for(int i=0; i < size; i++) {

    if((*p_bunch)[i].isLost()) {
      continue;
    }

    PAC::Position& p = (*p_bunch)[i].getPosition();
    if(fabs(p.getY())  > m_yMax)  m_yMax  = fabs(p.getY());
    if(fabs(p.getPY()) > m_pyMax) m_pyMax = fabs(p.getPY());
  } 

  // std::cout << "ctde viewer: update points " << std::endl;

}

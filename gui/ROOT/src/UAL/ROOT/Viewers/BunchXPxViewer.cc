#include <qapplication.h>

#include "UAL/Common/Def.hh"
#include "UAL/ROOT/Viewers/BunchXPxViewer.hh"


UAL::ROOT::BunchXPxViewer::BunchXPxViewer(UAL::QT::BasicPlayer* player, 
					  PAC::Bunch* bunch)
  : UAL::ROOT::BasicViewer()
{

  p_player     = player;
  p_bunch      = bunch;


  findLimits();

  m_xbins  = 100;
  m_pxbins = 100;
  xpx = new TH2F("xpx", "X-PX Phase Space", 
		 m_xbins,  -1.5*m_xMax,  +1.5*m_xMax, 
		 m_pxbins, -1.5*m_pxMax, +1.5*m_pxMax);

  xpx->Draw();
  xpx->SetBit(kCanDelete);

  xpx->SetOption("colz");
  xpx->GetXaxis()->SetTitle("x, m");
  xpx->GetYaxis()->SetTitle("px/p0");
  xpx->GetYaxis()->CenterTitle();

  updatePoints();

  viewer->GetCanvas()->cd();

  viewer->GetCanvas()->Modified();
  viewer->GetCanvas()->Update();
}


void UAL::ROOT::BunchXPxViewer::updateViewer(int turn)
{
  updatePoints();

  viewer->cd();

  viewer->GetCanvas()->Modified();
  viewer->GetCanvas()->Update();

}

void UAL::ROOT::BunchXPxViewer::closeEvent(QCloseEvent* ce)
{

  ce->accept();
  p_player->removeViewer("UAL::ROOT::BunchXPxViewer");

}

void UAL::ROOT::BunchXPxViewer::updatePoints()
{
  if(!p_bunch) return;

  int size = p_bunch->size();

  xpx->Reset("");

  for(int i=0; i < size; i++) {

    if((*p_bunch)[i].isLost()) {
      continue;
    }

    PAC::Position& p = (*p_bunch)[i].getPosition();
    xpx->Fill(p.getX(), p.getPX());
  } 

  // std::cout << "ctde viewer: update points " << std::endl;

}

void UAL::ROOT::BunchXPxViewer::findLimits()
{
  m_xMax  = +1.e-20;
  m_pxMax = +1.e-20;

  if(!p_bunch) return;

  int size = p_bunch->size();

  for(int i=0; i < size; i++) {

    if((*p_bunch)[i].isLost()) {
      continue;
    }

    PAC::Position& p = (*p_bunch)[i].getPosition();
    if(fabs(p.getX())  > m_xMax)  m_xMax  = fabs(p.getX());
    if(fabs(p.getPX()) > m_pxMax) m_pxMax = fabs(p.getPX());
  } 

  // std::cout << "ctde viewer: update points " << std::endl;

}

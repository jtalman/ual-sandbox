
#include <iostream>
#include <sstream>
#include "UAL/ROOT/Viewers/MountainRangeViewer.hh"


UAL::ROOT::MountainRangeViewer::MountainRangeViewer(UAL::QT::BasicPlayer* player, 
						    PAC::Bunch* bunch)
  : UAL::ROOT::BasicViewer()
{

  p_player     = player;
  p_bunch      = bunch;

  p_separatrix = &UAL::SeparatrixCalculator::getInstance(); 

  m_nbins  = 100;
  m_nsteps = 100;

  double ctMax = p_separatrix->getSumL()/p_separatrix->getRFCavity().getHarmon()/2.;

  int fprint = p_player->getFprint();

  std::stringstream my_stringstream;
  my_stringstream <<  fprint;
  std::string strFprint = my_stringstream.str();
  strFprint += " turns";
  
  m_mr = new TH2F("mr", "Mountain Range Plot", 
		  m_nbins, -ctMax, +ctMax,
  		  m_nsteps, 0, m_nsteps);
  m_mr->SetBit(kCanDelete);
  m_mr->Draw("colz");
  m_mr->GetXaxis()->SetTitle("ct, m");
  m_mr->GetYaxis()->SetTitle(strFprint.c_str());
  m_mr->Draw("colz"); 

  updatePoints(0);

  viewer->cd(); 
  viewer->GetCanvas()->Modified();
  viewer->GetCanvas()->Update();
}

void UAL::ROOT::MountainRangeViewer::updateViewer(int turn)
{

  if(turn == 0){
    std::cout << "Before Reset " << std::endl;
    m_mr->Reset("");
    std::cout << "After Reset " << std::endl;
  }

  updatePoints(turn/m_fprint);
  viewer->cd(); 

  viewer->GetCanvas()->Modified();
  viewer->GetCanvas()->Update();

}

void UAL::ROOT::MountainRangeViewer::closeEvent(QCloseEvent* ce)
{

  ce->accept();
  p_player->removeViewer("UAL::ROOT::MountainRangeViewer");

}

void UAL::ROOT::MountainRangeViewer::updatePoints(int step)
{
  if(!p_bunch) return;

  for(int i=0; i < p_bunch->size(); i++) {

    if((*p_bunch)[i].isLost()) {
      continue;
    }

    PAC::Position& p = (*p_bunch)[i].getPosition();
    m_mr->Fill(p.getCT(), step);
  } 

  // std::cout << "mp viewer: update points " << std::endl;

}


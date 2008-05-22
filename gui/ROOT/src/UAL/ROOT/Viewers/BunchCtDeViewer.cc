#include <qapplication.h>

#include "UAL/Common/Def.hh"
#include "UAL/ROOT/Viewers/BunchCtDeViewer.hh"


UAL::ROOT::BunchCtDeViewer::BunchCtDeViewer(UAL::QT::BasicPlayer* player, 
					    PAC::Bunch* bunch)
  : UAL::ROOT::BasicViewer()
{

  p_player     = player;
  p_bunch      = bunch;
 
  p_separatrix = &UAL::SeparatrixCalculator::getInstance(); 

  double phaseS = 360.0*p_separatrix->getRFCavity().getLag();
  double deMax  = p_separatrix->getDeMax();

  // std::cout << "deMax = " << deMax << std::endl;
  if(deMax == 0.0) deMax = 1.0e-3;

  m_ctbins = 100;
  m_debins = 100;
  ctde = new TH2F("ctde", "Longitudinal Phase Space", 
  		  m_ctbins, phaseS - 180., phaseS + 180. , 
  		  m_debins, -deMax, +deMax);

  ctde->Draw();
  ctde->SetBit(kCanDelete);

  ctde->SetOption("colz");
  ctde->GetXaxis()->SetTitle("phase, degrees");
  ctde->GetYaxis()->SetTitle("de/p0");
  ctde->GetYaxis()->CenterTitle();

  tContour = new TGraph(m_ctbins);
  tContour->SetLineStyle(1);
  tContour->SetLineColor(8);
  tContour->SetBit(kCanDelete);

  bContour = new TGraph(m_ctbins);
  bContour->SetLineStyle(1);
  bContour->SetLineColor(8);
  bContour->SetBit(kCanDelete);

  updatePoints();
  // updateSeparatrix();

  viewer->GetCanvas()->cd();

  tContour->Draw();
  bContour->Draw();

  // viewer->Refresh();
  // update();

  viewer->GetCanvas()->Modified();
  viewer->GetCanvas()->Update();
}


void UAL::ROOT::BunchCtDeViewer::updateViewer(int turn)
{

  updatePoints();
  // updateSeparatrix();

  viewer->cd();

  // ctde->Draw("AP");

  // tContour->Draw();
  // bContour->Draw();

  // viewer->Refresh();
  viewer->GetCanvas()->Modified();
  viewer->GetCanvas()->Update();

  // update();
}

void UAL::ROOT::BunchCtDeViewer::closeEvent(QCloseEvent* ce)
{

  ce->accept();
  // p_page->closePlot();
  p_player->removeViewer("UAL::ROOT::BunchCtDeViewer");

}

void UAL::ROOT::BunchCtDeViewer::updateSeparatrix()
{
  if(!p_separatrix) return;

  p_separatrix->setBeamAttributes(p_bunch->getBeamAttributes());
  p_separatrix->calculate();

  std::vector<double>& phases = p_separatrix->m_phases;
  std::vector<double>& des = p_separatrix->m_des;

  int size = phases.size();
  if(size > m_ctbins) size = m_ctbins;   
    
  tContour->Set(size);
  bContour->Set(size); 

  double rad2degree = 180.0/UAL::pi;

  for(int i = 0; i < size; i++){
    tContour->SetPoint(i, phases[i]*rad2degree, +des[i]);
    bContour->SetPoint(i, phases[i]*rad2degree, -des[i]);
  }

 
}

void UAL::ROOT::BunchCtDeViewer::updatePoints()
{
  if(!p_bunch) return;

  PAC::BeamAttributes& ba = p_bunch->getBeamAttributes();

  double charge = ba.getCharge();
  double e      = ba.getEnergy();
  double m      = ba.getMass();
  double v0byc2 = (e*e - m*m)/e/e;
  double v0byc  = sqrt(v0byc2);

  double harmon = p_separatrix->getRFCavity().getHarmon();
  double phaseS = 360.0*p_separatrix->getRFCavity().getLag();
  double ct2phi = (360.0*harmon)*(v0byc/p_separatrix->getSumL());

  int size = p_bunch->size();

  ctde->Reset("");

  for(int i=0; i < size; i++) {

    if((*p_bunch)[i].isLost()) {
      continue;
    }

    PAC::Position& p = (*p_bunch)[i].getPosition();
    ctde->Fill(-p.getCT()*ct2phi + phaseS, p.getDE());
  } 

  if(!p_separatrix) return;

  p_separatrix->setBeamAttributes(p_bunch->getBeamAttributes());
  p_separatrix->calculate();

  std::vector<double>& phases = p_separatrix->m_phases;
  std::vector<double>& des = p_separatrix->m_des;

  size = phases.size();
  if(size > m_ctbins) size = m_ctbins;   
    
  tContour->Set(size);
  bContour->Set(size);  

  double rad2degree = 180.0/UAL::pi;

  for(int i = 0; i < size; i++){
    tContour->SetPoint(i, phases[i]*rad2degree, +des[i]);
    bContour->SetPoint(i, phases[i]*rad2degree, -des[i]);
  }


  // std::cout << "ctde viewer: update points " << std::endl;

}

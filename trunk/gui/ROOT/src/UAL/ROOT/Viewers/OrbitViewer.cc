#include <fstream>

#include <qapplication.h>
#include <qmenubar.h>
#include <qfiledialog.h>
#include <qmessagebox.h>

#include "UAL/Common/Def.hh"
#include "UAL/ROOT/Viewers/OrbitViewer.hh"


UAL::ROOT::OrbitViewer::OrbitViewer(UAL::QT::BasicPlayer* player,
				    std::vector<double>& atVector,
				    std::vector<PAC::Position>& orbitVector)
  : UAL::ROOT::BasicViewer()
{

  m_atVector    = atVector;
  m_orbitVector = orbitVector;

  // viewer->GetCanvas()->Divide(0, 2);

  p_player     = player;

  int size = atVector.size();

  int lineWidth = 1;

  TGraph* hOrbit = new TGraph(atVector.size());

  // hTwiss->SetTitle("Twiss, [m]");
  hOrbit->SetLineStyle(1);
  hOrbit->SetLineWidth(lineWidth);
  hOrbit->SetLineColor(2);
  // hTwiss->SetBit(kCanDelete);

  TGraph* vOrbit = new TGraph(atVector.size());
  vOrbit->SetLineStyle(1);
  vOrbit->SetLineWidth(lineWidth);
  vOrbit->SetLineColor(4);
  // vTwiss->SetBit(kCanDelete);

  legOrbit = new TLegend(0.6, 0.69, 0.89, 0.89);
  legOrbit->AddEntry(hOrbit, "Horizontal", "l");
  legOrbit->AddEntry(vOrbit, "Vertical", "l");
  legOrbit->SetBit(kCanDelete);

  for(unsigned int i=0; i < m_atVector.size(); i++){

    double x = m_orbitVector[i].getX();
    hOrbit->SetPoint(i, m_atVector[i], x);

    double y = m_orbitVector[i].getY();
    vOrbit->SetPoint(i, m_atVector[i], y);
  }

  mgOrbit = new TMultiGraph();
  mgOrbit->Add(hOrbit);
  mgOrbit->Add(vOrbit);

  viewer->cd();

  mgOrbit->Draw("AL");
  mgOrbit->SetTitle("Orbit, [m]");
  mgOrbit->GetXaxis()->SetTitle("s, m");
  mgOrbit->SetBit(kCanDelete);
  legOrbit->Draw();
 
  viewer->GetCanvas()->Modified();
  viewer->GetCanvas()->Update();
}


void UAL::ROOT::OrbitViewer::closeEvent(QCloseEvent* ce)
{
  ce->accept();
  p_player->removeViewer("UAL::ROOT::OrbitViewer");

}





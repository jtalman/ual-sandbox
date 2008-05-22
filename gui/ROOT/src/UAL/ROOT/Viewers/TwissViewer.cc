#include <fstream>

#include <qapplication.h>
#include <qmenubar.h>
#include <qfiledialog.h>
#include <qmessagebox.h>

#include "UAL/Common/Def.hh"
#include "UAL/ROOT/Viewers/TwissViewer.hh"


UAL::ROOT::TwissViewer::TwissViewer(UAL::QT::BasicPlayer* player,
				    std::vector<double>& atVector,
				    std::vector<PacTwissData>& twissVector)
  : UAL::ROOT::BasicViewer()
{

  updateMenu();

  m_atVector    = atVector;
  m_twissVector = twissVector;

  viewer->GetCanvas()->Divide(0, 2);

  p_player     = player;

  int size = atVector.size();
  double maxBeta = findMaxBeta(twissVector);

  int lineWidth = 1;

  TGraph* hTwiss = new TGraph(atVector.size());

  // hTwiss->SetTitle("Twiss, [m]");
  hTwiss->SetLineStyle(1);
  hTwiss->SetLineWidth(lineWidth);
  hTwiss->SetLineColor(2);
  // hTwiss->SetBit(kCanDelete);

  TGraph* vTwiss = new TGraph(atVector.size());
  vTwiss->SetLineStyle(1);
  vTwiss->SetLineWidth(lineWidth);
  vTwiss->SetLineColor(4);
  // vTwiss->SetBit(kCanDelete);

  legTwiss = new TLegend(0.6, 0.69, 0.89, 0.89);
  legTwiss->AddEntry(hTwiss, "Horizontal", "l");
  legTwiss->AddEntry(vTwiss, "Vertical", "l");
  legTwiss->SetBit(kCanDelete);


  hD = new TGraph(atVector.size());
  hD->SetTitle("Dispersion, [m]");
  hD->SetLineStyle(1);
  hD->SetLineWidth(lineWidth);
  hD->SetLineColor(2);
  hD->SetBit(kCanDelete);

  vD = new TGraph(atVector.size());
  vD->SetLineStyle(1);
  vD->SetLineWidth(lineWidth);
  vD->SetLineColor(4);
  vD->SetBit(kCanDelete);

  legD = new TLegend(0.6, 0.69, 0.89, 0.89);
  legD->AddEntry(hTwiss, "Horizontal", "l");
  legD->AddEntry(vTwiss, "Vertical", "l");
  legD->SetBit(kCanDelete);

  for(unsigned int i=0; i < m_atVector.size(); i++){

    double betax = m_twissVector[i].beta(0);
    hTwiss->SetPoint(i, m_atVector[i], betax);

    double betay = m_twissVector[i].beta(1);
    vTwiss->SetPoint(i, m_atVector[i], betay);

    hD->SetPoint(i, m_atVector[i], m_twissVector[i].d(0));
    vD->SetPoint(i, m_atVector[i], m_twissVector[i].d(1));
  }

  mgTwiss = new TMultiGraph();
  mgTwiss->Add(hTwiss);
  mgTwiss->Add(vTwiss);


  viewer->cd(1);

  mgTwiss->Draw("AL");
  mgTwiss->SetTitle("Twiss, [m]");
  mgTwiss->GetXaxis()->SetTitle("s, m");
  mgTwiss->SetBit(kCanDelete);
  legTwiss->Draw();

  /*
  std::cout << "max beta " << maxBeta << std::endl;
  hTwiss->GetHistogram()->GetYaxis()->SetLimits(0, maxBeta); 
  hTwiss->Draw("AL");
  hTwiss->GetXaxis()->SetTitle("s, m");
  hTwiss->GetHistogram()->GetYaxis()->SetLimits(0, maxBeta); 
  hTwiss->Draw("AL");
  // hTwiss->GetYaxis()->SetTitle("m");
  // hTwiss->GetYaxis()->CenterTitle();
  vTwiss->Draw("L");
  legTwiss->Draw();
  */

  viewer->cd(2);
  hD->Draw("AL");
  hD->GetXaxis()->SetTitle("s, m");
  // hD->GetYaxis()->SetTitle("m");
  // hD->GetYaxis()->CenterTitle();
  vD->Draw("L");
  legD->Draw();

 
  viewer->GetCanvas()->Modified();
  viewer->GetCanvas()->Update();
}

double UAL::ROOT::TwissViewer::findMaxBeta(std::vector<PacTwissData>& twissVector)
{
  double maxBetax = 0.0, maxBetay = 0.0;

  for(unsigned int i=0; i < twissVector.size(); i++){

    double betax = twissVector[i].beta(0);
    if(betax > maxBetax) maxBetax = betax;

    double betay =twissVector[i].beta(1);
    if(betay > maxBetay) maxBetay = betay;
  }

  return std::max(maxBetax, maxBetay);
}



void UAL::ROOT::TwissViewer::closeEvent(QCloseEvent* ce)
{
  ce->accept();
  p_player->removeViewer("UAL::ROOT::TwissViewer");

}

void UAL::ROOT::TwissViewer::updateMenu()
{
  writeToAct = new QAction(tr("Write &To..."), tr("Write &To..."), 0, this);
  writeToAct->addTo(fileMenu);
  connect(writeToAct, SIGNAL(activated()), this, SLOT(writeTo()));
}

bool UAL::ROOT::TwissViewer::writeTo()
{
  QString fileName = QFileDialog::getSaveFileName(".", tr("All files (*.*)"), this);
  if (fileName.isEmpty())
    return false;

  if (QFile::exists(fileName)) {
    int ret = QMessageBox::warning(this, tr("Warning"),
	   tr("File %1 already exists.\n"
	      "Do you want to overwrite it?")
               .arg(QDir::convertSeparators(fileName)),
	       QMessageBox::Yes | QMessageBox::Default,
               QMessageBox::No | QMessageBox::Escape);
    if (ret == QMessageBox::No) return true;
  }
    
  if (!fileName.isEmpty()) writeToFile(fileName);
  return true;
}

void UAL::ROOT::TwissViewer::writeToFile(const QString& fileName)
{

  std::ofstream out(fileName);

  std::vector<std::string> columns(11);
  columns[0]  = "#";
  columns[1]  = "name";
  columns[2]  = "suml";
  columns[3]  = "betax";
  columns[4]  = "alfax";
  columns[5]  = "mux";
  columns[6]  = "dx";
  columns[7]  = "betay";
  columns[8]  = "alfay";
  columns[9]  = "muy";
  columns[10] = "dy";

  char endLine = '\0';

  double twopi = 2.0*UAL::pi;


  out << "------------------------------------------------------------";
  out << "------------------------------------------------------------" << std::endl; 

  char line[200];
  sprintf(line, "%-5s %-10s   %-15s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s%c", 
	columns[0].c_str(),  columns[1].c_str(), columns[2].c_str(), columns[3].c_str(),  
	columns[4].c_str(),
	columns[5].c_str(), columns[6].c_str(), columns[7].c_str(), columns[8].c_str(),  
	columns[9].c_str(), columns[10].c_str(), endLine);
  out << line << std::endl;

  out << "------------------------------------------------------------";
  out << "------------------------------------------------------------" << std::endl; 

  std::string bName;

  for(unsigned int i=0; i < m_atVector.size(); i++){
    sprintf(line, "%5d %-10s %15.7e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e%c", 
	    i, bName.c_str(), m_atVector[i], 
	    m_twissVector[i].beta(0), m_twissVector[i].alpha(0), 
	    m_twissVector[i].mu(0)*twopi, m_twissVector[i].d(0),
	    m_twissVector[i].beta(1), m_twissVector[i].alpha(1), 
	    m_twissVector[i].mu(1)*twopi, m_twissVector[i].d(1), endLine);
    out << line << std::endl;
  }

  out.close();
}




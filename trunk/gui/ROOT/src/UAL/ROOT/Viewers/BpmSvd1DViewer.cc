
#include <cmath>

#include <qapplication.h>
#include <qpushbutton.h>
#include <qspinbox.h>
#include <qslider.h>
#include <qtabwidget.h> 
#include <qtable.h>
#include <qlayout.h>
#include <qvbox.h>
#include <rfftw.h>

#include "Optics/PacTMap.h"
#include "Optics/PacChromData.h"
#include "Main/Teapot.h"
#include "UAL/UI/OpticsCalculator.hh"

// #include "timer.h"

#include "AIM/BPM/Monitor.hh"
#include "AIM/BPM/MonitorCollector.hh"

#include "UAL/ROOT/Viewers/BpmSvd1DViewer.hh"

UAL::ROOT::BpmSvd1DViewer::BpmSvd1DViewer(UAL::QT::BasicPlayer* player, int plane)
  : UAL::ROOT::BasicViewer()
{
  // double t;
  // start_ms();

  m_plane = plane;

  std::map<int, AIM::Monitor*> bpms = AIM::MonitorCollector::getInstance().getAllData();
  m_nbpms  = bpms.size();
  m_nturns = bpms.begin()->second->getData().size();

  int maxStep = 3*(m_nbpms/10) + 10;
  pd = new QProgressDialog("Operation in progress.","Cancel", maxStep);
  pd->setAutoClose(true);
  connect(pd,SIGNAL(canceled()), this,SLOT(stopSVD()));

  int w = 500;
  int h = 400;
  int x = QApplication::desktop()->width()/2 - w/2;
  int y = QApplication::desktop()->height()/2 - h/2;

  setGeometry(x, y, w, h);

  tabView = new Svd1DViewUI(m_frame);

  QObject::connect(tabView->tabWidget, SIGNAL(currentChanged(QWidget*)),
		   this, SLOT(updateViewer(QWidget*)));
  QObject::connect(tabView->modeButton, SIGNAL( clicked() ),
		   this, SLOT( updateSvdGraphs() ));

  QVBoxLayout* twissLayout = new QVBoxLayout(tabView->twissFrame); 
  twissViewer = new TQtWidget(tabView->twissFrame,"EmbeddedCanvas");
  twissLayout->addWidget(twissViewer);

  QVBoxLayout* svdLayout = new QVBoxLayout(tabView->svdFrame);
  svdViewer   = new TQtWidget(tabView->svdFrame,"EmbeddedCanvas");
  svdLayout->addWidget(svdViewer);

  p_player   = player;

  isDone = false;

  // t = (end_ms());
  // std::cout << "createTwissGraphs: time  = " << t << " ms" << endl;

  createTwissGraphs();

  // t = (end_ms());
  // std::cout << "createSvdGraphs: time  = " << t << " ms" << endl;

  createSvdGraphs();

  // t = (end_ms());
  // std::cout << "startSVD: time  = " << t << " ms" << endl;

  m_worker.setViewer(this);
  m_worker.setMaxStep(maxStep);
  m_worker.startRun();

  // t = (end_ms());
  // std::cout << "isDone: time  = " << t << " ms" << endl;

}

void UAL::ROOT::BpmSvd1DViewer::createTwissGraphs()
{
  twissViewer->GetCanvas()->Divide(0, 2);

  createDesignTwissGraphs();
  createSvdTwissGraphs();

  legPhase = new TLegend(0.6, 0.69, 0.89, 0.89);
  legPhase->AddEntry(dPhaseGraph, "Design", "lp");
  legPhase->AddEntry(mPhaseGraph, "PCA", "p");
  legPhase->SetBit(kCanDelete);

  legBeta = new TLegend(0.6, 0.69, 0.89, 0.89);
  legBeta->AddEntry(dBetaGraph, "Design", "lp");
  legBeta->AddEntry(mBetaGraph, "PCA", "p");
  legBeta->SetBit(kCanDelete);

  viewer->GetCanvas()->Modified();
  viewer->GetCanvas()->Update();

}

void UAL::ROOT::BpmSvd1DViewer::createSvdTwissGraphs()
{

  int nbpms = m_nbpms;
  if(nbpms == 0) return;
 
  mPhaseGraph = new TGraph(nbpms); 
  // mPhaseGraph->SetTitle("Phases");
  mPhaseGraph->SetBit(kCanDelete);
  mPhaseGraph->SetMarkerStyle(7);
  mPhaseGraph->SetMarkerColor(2);

  mBetaGraph = new TGraph(nbpms); 
  // mBetaGraph->SetTitle("Beta");
  mBetaGraph->SetBit(kCanDelete);
  mBetaGraph->SetMarkerStyle(7);
  mBetaGraph->SetMarkerColor(2);

}

void UAL::ROOT::BpmSvd1DViewer::createDesignTwissGraphs()
{

  int nbpms = m_nbpms;
  if(nbpms == 0) return;
 
  dPhaseGraph = new TGraph(nbpms); 
  // dPhaseGraph->SetTitle("Phases");
  dPhaseGraph->SetBit(kCanDelete);
  dPhaseGraph->SetMarkerStyle(7);
  dPhaseGraph->SetLineColor(31);

  dBetaGraph = new TGraph(nbpms); 
  // dBetaGraph->SetTitle("Beta");
  dBetaGraph->SetBit(kCanDelete);
  dBetaGraph->SetMarkerStyle(7);
  dBetaGraph->SetLineColor(31);
}


void UAL::ROOT::BpmSvd1DViewer::updateTwissGraphs()
{
  // std::cout << "update twiss viewer " << std::endl;

  int nbpms = m_nbpms;

  double svdMaxBeta;
  std::vector<double> svdPhases;
  std::vector<double> svdBetas;
  calculateSvdTwiss(svdPhases, svdBetas, svdMaxBeta);

  dPhaseGraph->Set(nbpms);
  mPhaseGraph->Set(nbpms);

  mPhaseGraph->SetPoint(0, 0, 0.0);
  // svdPhases[0] = 0.0;

  for(int ibpm = 1; ibpm < nbpms; ibpm++){
    double dPhase = m_designPhases[ibpm] - m_designPhases[ibpm-1];
    // std::cout << ibpm << " design: " << m_designPhases[ibpm] 
    // << " " << m_designPhases[ibpm-1] << std::endl;
    dPhaseGraph->SetPoint(ibpm, ibpm, dPhase);
    double svdPhase = svdPhases[ibpm] - svdPhases[ibpm-1];
    // std::cout << ibpm << " svd: " << svdPhases[ibpm] 
    // << " " << svdPhases[ibpm-1] << std::endl;
    if(svdPhase < 0.0) {
      svdPhase += UAL::pi;
    }
    // std::cout << ibpm << "delta design : " << dPhase 
    // << "svd : " << svdPhase << std::endl;
    if(svdPhase > UAL::pi/2.0) svdPhase = UAL::pi - svdPhase;
    mPhaseGraph->SetPoint(ibpm, ibpm, svdPhase);
  }

  dBetaGraph->Set(nbpms);
  mBetaGraph->Set(nbpms);
  for(int ibpm = 0; ibpm < nbpms; ibpm++){
    dBetaGraph->SetPoint(ibpm, ibpm, m_designBetas[ibpm]);
    mBetaGraph->SetPoint(ibpm, ibpm, svdBetas[ibpm]/svdMaxBeta);
    // std::cout << ibpm << " " 
    //	      << m_designBetas[ibpm] << " " 
    //	      << svdBetas[ibpm]/svdMaxBeta << std::endl;
  }

  twissViewer->cd(1);
  dPhaseGraph->Draw("ALP");
  dPhaseGraph->SetTitle("Phases");
  dPhaseGraph->GetXaxis()->SetTitle("BPM index");
  dPhaseGraph->GetXaxis()->SetLimits(0, nbpms); 
  mPhaseGraph->Draw("P");
  legPhase->Draw();

  twissViewer->cd(2);
  dBetaGraph->Draw("ALP"); 
  dBetaGraph->SetTitle("Beta");
  dBetaGraph->GetXaxis()->SetTitle("BPM index");
  dBetaGraph->GetXaxis()->SetLimits(0, nbpms);
  mBetaGraph->Draw("P");
  legBeta->Draw();

  twissViewer->GetCanvas()->cd();
  twissViewer->GetCanvas()->Modified();
  twissViewer->GetCanvas()->Update();
}

void UAL::ROOT::BpmSvd1DViewer::updateTable()
{

  // std::cout << "update table " << std::endl;

  tabView->svdTable->setNumRows(m_worker.w.size());
  for(unsigned int im = 0; im < m_worker.w.size(); im++) {
    tabView->svdTable->setText(im, 0, QString::number(m_worker.w[im]));
  } 
}


void UAL::ROOT::BpmSvd1DViewer::createSvdGraphs()
{
  svdViewer->GetCanvas()->Divide(0, 2);

  int nbpms = m_nbpms;
  if(nbpms == 0) return;
  
  int nturns = m_nturns;
  if(nturns == 0) return;  

  tabView->modeSpinBox->setMinValue(0);
  tabView->modeSpinBox->setMaxValue(nbpms-1);

  tabView->modeSlider->setMinValue(0);
  tabView->modeSlider->setMaxValue(nbpms-1);


  uSvdGraph = new TGraph(nturns); 
  uSvdGraph->SetTitle("Temporal vector");
  uSvdGraph->SetBit(kCanDelete);
  uSvdGraph->SetMarkerStyle(2);
  uSvdGraph->SetLineColor(31);

  vSvdGraph = new TGraph(nbpms); 
  vSvdGraph->SetTitle("Spatial vector");
  vSvdGraph->SetBit(kCanDelete);
  vSvdGraph->SetMarkerStyle(20);
  vSvdGraph->SetLineColor(31);

}

void UAL::ROOT::BpmSvd1DViewer::updateSvdGraphs()
{

  int mode = tabView->modeSpinBox->value(); 

  // std::cout << "update svd graphs, mode = " << mode << std::endl;

  int nturns = m_worker.u.size();
  uSvdGraph->Set(nturns);
  for(int it = 0; it < nturns; it++){
    uSvdGraph->SetPoint(it, it, m_worker.u[it][mode]);
  }

  int nbpms = m_worker.v.size();
  vSvdGraph->Set(nbpms);
  for(int ibpm = 0; ibpm < nbpms; ibpm++){
      vSvdGraph->SetPoint(ibpm, ibpm, m_worker.v[ibpm][mode]);
  }

  svdViewer->cd(1);
  uSvdGraph->Draw("AL");
  uSvdGraph->GetXaxis()->SetTitle("turn index");
  uSvdGraph->GetXaxis()->SetLimits(0, nturns); 

  svdViewer->cd(2);
  vSvdGraph->Draw("AL"); 
  vSvdGraph->GetXaxis()->SetTitle("BPM index");
  vSvdGraph->GetXaxis()->SetLimits(0, nbpms);

  svdViewer->GetCanvas()->cd();
  svdViewer->GetCanvas()->Modified();
  svdViewer->GetCanvas()->Update();
}

void UAL::ROOT::BpmSvd1DViewer::saveFile(const QString& fileName)
{  
  int pageIndex = tabView->tabWidget->currentPageIndex();
  std::cout << "page inde =" << pageIndex << std::endl;
  switch(pageIndex) {
  case 0: 
    twissViewer->GetCanvas()->SaveAs(fileName);
    break;
  case 1:
    break;
  case 2:
    svdViewer->GetCanvas()->SaveAs(fileName);
    break;
  default:
    break;
  } 
}

void UAL::ROOT::BpmSvd1DViewer::closeEvent(QCloseEvent* ce)
{
  ce->accept();
  if(m_plane == 0) {
    p_player->removeViewer("UAL::ROOT::BpmSvdXViewer");
  } else {
    p_player->removeViewer("UAL::ROOT::BpmSvdYViewer");
  }
}

void UAL::ROOT::BpmSvd1DViewer::startSVD()
{
}

void UAL::ROOT::BpmSvd1DViewer::stopSVD()
{
  m_worker.stopRun();
  close();
}

void UAL::ROOT::BpmSvd1DViewer::customEvent(QCustomEvent* customEvent)
{
  if((int) customEvent->type() != 65433) return;

  UAL::ROOT::SVDWorkerEvent* svdEvent = (UAL::ROOT::SVDWorkerEvent*) customEvent;
  pd->setProgress(svdEvent->step);
  // std::cout << "progress = " << pd->progress() 
  //	    << " total steps = " << pd->totalSteps() 
  //	    << std::endl;
  if(pd->progress() < 0) {
    isDone = true;
    calculateDesignTwiss();
    tabView->tabWidget->setCurrentPage(1);
    updateViewer(tabView->tabWidget->currentPage());
    this->show();
  }
}

void UAL::ROOT::BpmSvd1DViewer::updateViewer(QWidget*)
{
  int index = tabView->tabWidget->currentPageIndex();
  switch(index) {
  case 0: 
    updateTwissGraphs();
    break;
  case 1:
    updateTable();
    break;
  case 2:
    updateSvdGraphs();
    break;
  default:
    break;
  };
}

void UAL::ROOT::BpmSvd1DViewer::selectElements(std::vector<int>& elemVector)
{

  std::map<int, AIM::Monitor*> bpms = AIM::MonitorCollector::getInstance().getAllData();
  std::map<int, AIM::Monitor*>::iterator it;

  std::list<int> elemList;
  for(it = bpms.begin(); it != bpms.end(); it++){
    int index = it->second->getIndex();
    if(index < 0)  index = 0;
    elemList.push_back(index);
  }

  elemVector.resize(elemList.size());

  int counter = 0;
  std::list<int>::iterator ie;
  for(ie = elemList.begin(); ie != elemList.end(); ie++){
    elemVector[counter++] = *ie;
  } 

}

void UAL::ROOT::BpmSvd1DViewer::calculateSvdTwiss(std::vector<double>& phases,
					    std::vector<double>& betas,
					    double& maxBeta)
{

  int nbpms = m_worker.v.size();

  maxBeta = 0.0;
  phases.resize(nbpms);
  betas.resize(nbpms);

  double phase; 
  for(int ibpm = 0; ibpm < nbpms; ibpm++){
    betas[ibpm]  =  m_worker.w[0]*m_worker.w[0]*m_worker.v[ibpm][0]*m_worker.v[ibpm][0];
    betas[ibpm] +=  m_worker.w[1]*m_worker.w[1]*m_worker.v[ibpm][1]*m_worker.v[ibpm][1];
    if(betas[ibpm] > maxBeta) maxBeta = betas[ibpm]; 

    phase  = m_worker.w[1]*m_worker.v[ibpm][1];
    phase /= m_worker.w[0]*m_worker.v[ibpm][0];
    phase  = std::atan(phase);
    if(phase < 0.0) phase = UAL::pi + phase;
    phases[ibpm] = phase;

  }
  
}

void UAL::ROOT::BpmSvd1DViewer::calculateDesignTwiss()
{

  std::vector<int> elems;
  selectElements(elems);

  double maxBeta = 0.0;
  std::vector<PacTwissData> twissVector;
  twissVector.resize(elems.size());

  UAL::OpticsCalculator& optics = UAL::OpticsCalculator::getInstance();

  PAC::Position orbit;
  optics.m_teapot->clorbit(orbit, optics.m_ba);

  PacChromData chrom;
  optics.m_teapot->chrom(chrom, optics.m_ba, orbit);

  PacTwissData twiss = chrom.twiss();

  PacTMap map(6);
  map.refOrbit(orbit);
  optics.m_teapot->map(map, optics.m_ba, 1);

  int mltOrder =  map.mltOrder();
  map.mltOrder(1);


  double mux = 0.0, muy = 0.0;
  twiss.mu(0, mux);
  twiss.mu(1, muy);

  int i1 = 0, i2;
  int counter = 0;
  for(unsigned int it = 0; it < elems.size(); it++){

    i2 = elems[it];

    // std::cout << it << " " << "bpms  : (" << i1 << ", " << i2 << ")" << std::endl;

    PacTMap map(6);
    map.refOrbit(orbit);

    optics.m_teapot->trackMap(map, optics.m_ba, i1, i2);
    optics.m_teapot->trackTwiss(twiss, map);

    if((twiss.mu(0) - mux) < 0.0) twiss.mu(0, twiss.mu(0) + 2.0*UAL::pi);
    mux = twiss.mu(0);

    if((twiss.mu(1) - muy) < 0.0) twiss.mu(1, twiss.mu(1) + 2.0*UAL::pi);
    muy = twiss.mu(1);

    twissVector[counter++] = twiss;
    if(twiss.beta(0) > maxBeta) maxBeta = twiss.beta(0);
    optics.m_teapot->trackClorbit(orbit, optics.m_ba, i1, i2);

    i1 = i2;    
  }

  map.mltOrder(mltOrder);

  m_designPhases.resize(twissVector.size());
  m_designBetas.resize(twissVector.size());

  for(unsigned int it = 0; it < twissVector.size(); it++){
    if(m_plane == 0){
      m_designPhases[it] = twissVector[it].mu(0);
      m_designBetas[it] = twissVector[it].beta(0)/maxBeta;
    } else {
      m_designPhases[it] = twissVector[it].mu(1);
      m_designBetas[it] = twissVector[it].beta(1)/maxBeta;
    }
  }

}




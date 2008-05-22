#ifndef UAL_ROOT_MOUNTAIN_RANGE_VIEWER_HH
#define UAL_ROOT_MOUNTAIN_RANGE_VIEWER_HH

#include <qvbox.h>
#include <qlistview.h>
#include <qevent.h>

#include "TROOT.h"
#include "TApplication.h"
#include "TPad.h"
#include "TGQt.h"
#include "TSystem.h"
#include "TCanvas.h"
#include <TH1D.h>
#include <TMultiGraph.h>
#include "TH2F.h"
#include "TGraph2D.h"

#include "PAC/Beam/Bunch.hh"

#include "UAL/QT/Player/BasicPlayer.hh"
#include "UAL/QT/Player/SeparatrixCalculator.hh"
#include "UAL/ROOT/Viewers/BasicViewer.hh"

namespace UAL
{
 namespace ROOT {
  class MountainRangeViewer : public BasicViewer
  {

    Q_OBJECT

  public:

    /** Constructor */
    MountainRangeViewer(UAL::QT::BasicPlayer* player, 
			PAC::Bunch* bunch);

    void updateViewer(int turn);

    /** Closes window and clean up state */
    void closeEvent(QCloseEvent* ce);

    /** Updates points */
    void updatePoints(int step);

    // void setTurns(int turns);
    // void setFprint(int fprint);


  private:

    // Parent player
    UAL::QT::BasicPlayer* p_player;

    int m_nbins;
    int m_nsteps;

    // Mountain Range plot 
    TH2F * m_mr;
    
  private:
    
    PAC::Bunch* p_bunch;
    UAL::SeparatrixCalculator* p_separatrix;
  };
 }
}

#endif

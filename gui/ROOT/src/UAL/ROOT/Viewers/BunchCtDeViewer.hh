#ifndef UAL_ROOT_BUNCH_CT_DE_VIEWER_HH
#define UAL_ROOT_BUNCH_CT_DE_VIEWER_HH

#include <qvbox.h>
#include <qlistview.h>
#include <qevent.h>

#include "TROOT.h"
#include "TApplication.h"
#include "TPad.h"
#include "TGQt.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TH2F.h"
#include "TGraph.h"

#include "PAC/Beam/Bunch.hh"


#include "UAL/QT/Player/BasicPlayer.hh"
#include "UAL/QT/Player/SeparatrixCalculator.hh"
#include "UAL/ROOT/Viewers/BasicViewer.hh"

namespace UAL
{
 namespace ROOT {
  class BunchCtDeViewer : public BasicViewer
  {

    Q_OBJECT

  public:

    /** Constructor */
    BunchCtDeViewer(UAL::QT::BasicPlayer* player, 
		    PAC::Bunch* bunch);

    /** Updates */
    void updateViewer(int turn);

    /** Closes window and clean up state */
    void closeEvent(QCloseEvent* ce);

    /** Updates points */
    void updatePoints();
    void updateSeparatrix();


  private:

    // Parent player
    UAL::QT::BasicPlayer* p_player;

    // ct-de  plot
    int m_ctbins, m_debins;
    TH2F* ctde;
    // TGraph* ctde;

    // separatrix
    TGraph* tContour;
    TGraph* bContour;   

  private:
    
    PAC::Bunch* p_bunch;
    UAL::SeparatrixCalculator* p_separatrix;
  };
 }
}

#endif

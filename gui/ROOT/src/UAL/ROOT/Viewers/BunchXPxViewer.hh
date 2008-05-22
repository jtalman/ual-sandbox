#ifndef UAL_ROOT_BUNCH_X_PX_VIEWER_HH
#define UAL_ROOT_BUNCH_X_PX_VIEWER_HH

#include <qvbox.h>
#include <qlistview.h>
#include <qevent.h>

#include "TROOT.h"
#include "TApplication.h"
#include "TPad.h"
#include "TGQt.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TQtWidget.h"
#include "TH2F.h"
#include "TGraph.h"

#include "PAC/Beam/Bunch.hh"

#include "UAL/QT/Player/BasicPlayer.hh"
#include "UAL/ROOT/Viewers/BasicViewer.hh"

namespace UAL
{
 namespace ROOT {
  class BunchXPxViewer : public BasicViewer
  {

    Q_OBJECT

  public:

    /** Constructor */
    BunchXPxViewer(UAL::QT::BasicPlayer* player, PAC::Bunch* bunch);

    /** Processes the event issued by tracker */
    void updateViewer(int turn);

    /** Closes window and clean up state */
    void closeEvent(QCloseEvent* ce);

  protected:

    /** Updates points */
    void updatePoints();


  private:

    // Parent player
    UAL::QT::BasicPlayer* p_player;

    // x-px  plot
    double m_xMax, m_pxMax;
    int m_xbins, m_pxbins;
    TH2F* xpx;

  private:
    
    PAC::Bunch* p_bunch;

  private:

    void findLimits();

  };
 }
}

#endif

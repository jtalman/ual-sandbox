#ifndef UAL_ROOT_ORBIT_VIEWER_HH
#define UAL_ROOT_ORBIT_VIEWER_HH

#include <vector>
#include <list>

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
#include "TLegend.h"
#include "TMultiGraph.h"

#include "PAC/Beam/Position.hh"

#include "UAL/QT/Player/BasicPlayer.hh"
#include "UAL/ROOT/Viewers/BasicViewer.hh"

namespace UAL
{
  namespace ROOT {

  /** Viewer of the closed orbit */
  class OrbitViewer : public BasicViewer
  {

    Q_OBJECT

  public:

    /** Constructor 
     * (position "at", closed orbit,  and TwissData will be combined into a new container)
     */
    OrbitViewer(UAL::QT::BasicPlayer* player,
		std::vector<double>& atVector,
		std::vector<PAC::Position>& orbitVector);

    /** Closes window and clean up state */
    void closeEvent(QCloseEvent* ce);

  protected:

    /** pointer to the application's main window */
    UAL::QT::BasicPlayer* p_player;

    TMultiGraph* mgOrbit;

    /** legend of Orbit plot */
    TLegend* legOrbit;

  private:

    std::vector<double> m_atVector;
    std::vector<PAC::Position> m_orbitVector; 

   };
  }
}

#endif

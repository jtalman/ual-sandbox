#ifndef UAL_ROOT_TWISS_VIEWER_HH
#define UAL_ROOT_TWISS_VIEWER_HH

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

#include "Optics/PacTwissData.h"

#include "UAL/QT/Player/BasicPlayer.hh"
#include "UAL/ROOT/Viewers/BasicViewer.hh"

namespace UAL
{
  namespace ROOT {

  /** Viewer of the lattice functions */
  class TwissViewer : public BasicViewer
  {

    Q_OBJECT

  public:

    /** Constructor 
     * (position "at" and TwissData will be combined into a new container)
     */
    TwissViewer(UAL::QT::BasicPlayer* player,
		std::vector<double>& atVector,
		std::vector<PacTwissData>& twissVector);

    /** Closes window and clean up state */
    void closeEvent(QCloseEvent* ce);

  protected:

    /** pointer to the application's main window */
    UAL::QT::BasicPlayer* p_player;

    TMultiGraph* mgTwiss;

    /** plot with  horizontal twiss functions */
    // TGraph* hTwiss;

    /** plot with vertical twiss functions */
    // TGraph* vTwiss; 

    /** legend of Twiss plot */
    TLegend* legTwiss;

    /** plot with  horizontal dispersion */
    TGraph* hD;

    /** plot with vertical dispersion */
    TGraph* vD; 

    /** legend of Dispersion plot */
    TLegend* legD;

  private:

    double findMaxBeta(std::vector<PacTwissData>& twissVector);

    /** Extends a file menu with the "Write To" action */
    void updateMenu();

    /** Writes twiss data into the specified file */
    void writeToFile(const QString& fileName);

    /** Action which is responsible for writing to a file */
    QAction *writeToAct; 

  private:

    std::vector<double> m_atVector;
    std::vector<PacTwissData> m_twissVector; 

  private slots:

    /** Qt slot associated with writeToAct action */
    bool writeTo(); 

   };
  }
}

#endif

#ifndef UAL_QT_BPM_SVD_1D_VIEWER_HH
#define UAL_QT_BPM_SVD_1D_VIEWER_HH

#include <vector>

#include <qvbox.h>
#include <qlistview.h>
#include <qevent.h>
#include <qprogressdialog.h>

#include "TROOT.h"
#include "TApplication.h"
#include "TPad.h"
#include "TGQt.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TQtWidget.h"
#include "TGraph.h"
#include "TLegend.h"

#include "PAC/Beam/Position.hh"

#include "SVDWorker.hh"
#include "Svd1DViewUI.hh"

#include "UAL/QT/Player/BasicPlayer.hh"
#include "UAL/ROOT/Viewers/BasicViewer.hh"


namespace UAL
{
 namespace ROOT {
  class BpmSvd1DViewer : public UAL::ROOT::BasicViewer
  {

    Q_OBJECT

  public:

    /** Constructor */
    BpmSvd1DViewer(UAL::QT::BasicPlayer* player, int plane = 0);

    /** Closes window and clean up state */
    void closeEvent(QCloseEvent* ce);

    /** Processes an event posted by the update method */
    void customEvent(QCustomEvent* customEvent);

  public slots:

     void startSVD();
     void stopSVD();
     void updateViewer(QWidget*);
     void updateSvdGraphs();

  protected:

    /** Save the ROOT canvas into the specified file */
    virtual void saveFile(const QString& fileName);

  protected:

    /** Parent player */
    UAL::QT::BasicPlayer* p_player; 

    /** Progress dialog */
    QProgressDialog* pd;

    /** SVD calculator */
    SVDWorker m_worker; 

  protected:

    void createTwissGraphs();
    void createDesignTwissGraphs();
    void createSvdTwissGraphs();

    void updateTwissGraphs();

    void updateTable();

    void createSvdGraphs();

    void selectElements(std::vector<int>& elemVector);
    void calculateDesignTwiss();
    void calculateSvdTwiss(std::vector<double>& phases, 
			   std::vector<double>& betas,
			   double& maxBeta);

  protected:

    /** Plane. 0 - horizontal, 1 - vertical */
    int m_plane; 

    int m_nbpms;
    int m_nturns;

  protected:

    std::vector<double> m_designPhases;
    std::vector<double> m_designBetas;    

  protected:

    bool isDone;

  protected:

    Svd1DViewUI* tabView;

  protected:

    TQtWidget *twissViewer; 

    TLegend* legBeta;
    TGraph* dPhaseGraph;
    TGraph* dBetaGraph;

    TLegend* legPhase;
    TGraph* mPhaseGraph;
    TGraph* mBetaGraph;

  protected:

    TQtWidget *svdViewer; 

    TGraph* uSvdGraph;
    TGraph* vSvdGraph;
   
  };
 }
}

#endif

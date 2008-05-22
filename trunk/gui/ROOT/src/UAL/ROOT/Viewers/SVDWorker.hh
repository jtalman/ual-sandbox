#ifndef UAL_ROOT_SVD_WORKER_HH
#define UAL_ROOT_SVD_WORKER_HH

#include <vector>

#include <qevent.h>
#include <qthread.h>
#include <qprogressdialog.h>
#include <qtimer.h>

namespace UAL
{
 namespace ROOT {

  class BpmSvd1DViewer;

  class SVDWorkerEvent : public QCustomEvent 
  {
  public:

    SVDWorkerEvent(int step);
    int step;
  };

  class SVDWorker : public QThread
  {

  public:

    /** Constructor */
    SVDWorker();

    void setViewer(BpmSvd1DViewer* viewer) { p_viewer = viewer; }
    void setMaxStep(int maxStep) { m_maxStep = maxStep; }

    void run();

    void startRun();
    void stopRun();

  public:

    std::vector< std::vector<double> > u; // (nturns);
    std::vector< double> w; // (nbpms);
    std::vector< std::vector<double> > v; // (nbpms);    

  protected:

    BpmSvd1DViewer* p_viewer;

    int m_step;
    int m_maxStep;

    QMutex mutex;
    volatile bool stopped;

  protected:

    void svbksb(const std::vector< std::vector<double> >& u, 
		const std::vector<double>& w, 
		const std::vector< std::vector<double> >& v, 
		const std::vector<double>& b, 
		std::vector<double>& x);

    void svdcmp(std::vector< std::vector<double> >& a, 
		std::vector<double>& w, 
		std::vector< std::vector<double> >& v);

    double PYTHAG(double a, double b);
    double SIGN(double a, double b);

  };

  inline double UAL::ROOT::SVDWorker::SIGN(double a, double b) 
  { 
    return b >= 0.0 ? fabs(a) : -fabs(a); 
  }

  inline double UAL::ROOT::SVDWorker::PYTHAG(double a, double b)
  {
    a = fabs(a);
    b = fabs(b);
    if(a>b) {
        double c=b/a;
        return a*sqrt(1+c*c);
    }
    else if(b!=0) {
        double c=a/b;
        return b*sqrt(1+c*c);
    }
    else
        return 0.0;
  }
 }
}

#endif

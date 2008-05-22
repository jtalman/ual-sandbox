
#include <qapplication.h>
#include <qprogressdialog.h>
#include <qtimer.h>

#include "AIM/BPM/Monitor.hh"
#include "AIM/BPM/MonitorCollector.hh"

#include "UAL/ROOT/Viewers/BpmSvd1DViewer.hh"

UAL::ROOT::SVDWorkerEvent::SVDWorkerEvent(int s)
  : QCustomEvent(65433)
{
  step = s;
}

UAL::ROOT::SVDWorker::SVDWorker()
{
  m_step    = 0;
  m_maxStep = 0;
  stopped = false;
}

void UAL::ROOT::SVDWorker::startRun(){
  if(!running()) {
    start();
  }
}

void UAL::ROOT::SVDWorker::stopRun()
{
  mutex.lock();
  stopped = true;
  mutex.unlock();
}

void UAL::ROOT::SVDWorker::run()
{
  // double t;
  // start_ms();
  std::map<int, AIM::Monitor*> bpms = AIM::MonitorCollector::getInstance().getAllData();

  int nbpms = bpms.size();
  if(nbpms == 0) return;

  int nturns = bpms.begin()->second->getData().size();

  u.resize(nturns);
  w.resize(nbpms);
  v.resize(nbpms);

  for(int i = 0; i < nturns; i++) u[i].resize(nbpms);
  for(int i = 0; i < nbpms; i++) v[i].resize(nbpms);

  int ib = 0;
  std::map<int, AIM::Monitor*>::iterator ibpms;
  for(ibpms = bpms.begin(); ibpms != bpms.end(); ibpms++){

    std::list<PAC::Position>& tbt = ibpms->second->getData();

    int it = 0;
    std::list<PAC::Position>::iterator itbt;
    for(itbt = tbt.begin(); itbt != tbt.end(); itbt++){
      u[it][ib] = itbt->getX();
      it++;
    }
    ib++;
  }

  // t = (end_ms());
  // std::cout << "start svdcmp: time  = " << t << " ms" << endl;

  svdcmp(u, w, v);

  /*
  for(int j = 0; j < nbpms; j++) {
    std::cout << "w: " << j << " " << w[j] << std::endl;
  }
  */


}

void UAL::ROOT::SVDWorker::svbksb(const std::vector< std::vector<double> >& u, 
			    const std::vector<double>& w, 
			    const std::vector< std::vector<double> >& v, 
			    const std::vector<double>& b, 
			    std::vector<double>& x)
{
    const int m = b.size();
    const int n = w.size();
    int jj,j,i;
    std::vector<double> tmp(n);

    for(j=0;j<n;j++) {
        double s=0.0;
        if (w[j]) {
            for (i=0;i<m;i++)
                s += u[i][j]*b[i];
            s /= w[j];
        }
        tmp[j]=s;
    }
    for(j=0;j<n;j++) {
        double s=0.0;
        for (jj=0;jj<n;jj++)
            s += v[j][jj]*tmp[jj];
        x[j]=s;
    }
}

void UAL::ROOT::SVDWorker::svdcmp(std::vector< std::vector<double> >& a, 
			    std::vector<double>& w, 
			    std::vector< std::vector<double> >& v)
{
    const int m = a.size();
    const int n = v.size();

    if(m < n) {
      std::cout << "SVDCMP: you must augment A with extra zero rows" << std::endl;
      return;
    }

    QApplication::postEvent(p_viewer, new UAL::ROOT::SVDWorkerEvent(m_step++));
    mutex.lock();
    if(stopped) {
      stopped = false;
      mutex.unlock();
      return;
    }
    mutex.unlock();

    // double t = (end_ms());
    // std::cout << "in svdcmp: time  = " << t << " ms" << endl;

    int flag,i,its,j,jj,k,l,nm;
    double c,f,h,s,x,y,z;
    double anorm=0.0,g=0.0,scale=0.0;

    std::vector<double> rv1(n);

    // std::cout << "Householder reduction to biodiagonal form" << std::endl;
    // t = (end_ms());
    // std::cout << "time  = " << t << " ms" << endl;
 
    QApplication::postEvent(p_viewer, new UAL::ROOT::SVDWorkerEvent(m_step++));
    mutex.lock();
    if(stopped) {
      stopped = false;
      mutex.unlock();
      return;
    }
    mutex.unlock();

    for (i=0;i<n;i++) {
      if((i/10)*10 == i) {
	QApplication::postEvent(p_viewer, new UAL::ROOT::SVDWorkerEvent(m_step++));
	mutex.lock();
	if(stopped) {
	  stopped = false;
	  mutex.unlock();
	  return;
	}
	mutex.unlock();
      }
      // t = (end_ms());
      // std::cout << "i = " << i << " : time  = " << t << " ms" << endl;
        l=i+1;
        rv1[i]=scale*g;
        g=s=scale=0.0;
        if (i < m) {
            for (k=i; k < m; k++) scale += fabs(a[k][i]);
            if (scale) {
                for (k=i; k < m; k++) {
                    a[k][i] /= scale;
                    s += a[k][i]*a[k][i];
                }
                f=a[i][i];
                g = -SIGN(sqrt(s),f);
                h=f*g-s;
                a[i][i]=f-g;
                if (i != n-1) {
                    for (j=l; j < n; j++) {
                        for (s=0.0, k= i; k < m; k++) s += a[k][i]*a[k][j];
                        f=s/h;
                        for (k=i; k < m; k++) a[k][j] += f*a[k][i];
                    }
                }
                for (k=i; k < m; k++) a[k][i] *= scale;
            }
        }
        w[i]=scale*g;
        g=s=scale=0.0;
        if (i < m && i != n-1) {
            for (k=l; k < n; k++) scale += fabs(a[i][k]);
            if (scale) {
                for (k=l; k < n;k++) {
                    a[i][k] /= scale;
                    s += a[i][k]*a[i][k];
                }
                f=a[i][l];
                g = -SIGN(sqrt(s),f);
                h=f*g-s;
                a[i][l]=f-g;
                for (k=l;k<n;k++) rv1[k]=a[i][k]/h;
                if (i != m-1) {
                    for (j=l;j<m;j++) {
                        for (s=0.0,k=l;k<n;k++) s += a[j][k]*a[i][k];
                        for (k=l;k<n;k++) a[j][k] += s*rv1[k];
                    }
                }
                for (k=l;k<n;k++) a[i][k] *= scale;
            }
        }
        anorm=std::max(anorm,(fabs(w[i])+fabs(rv1[i])));
    }

    // std::cout << "Accumulation of right-hand transformations" << std::endl;
    // t = (end_ms());
    // std::cout << "time  = " << t << " ms" << endl;

    QApplication::postEvent(p_viewer, new UAL::ROOT::SVDWorkerEvent(m_step++));
    mutex.lock();
    if(stopped) {
      stopped = false;
      mutex.unlock();
      return;
    }
    mutex.unlock();

    for (i=n-1; i >= 0;i--) {
      // t = (end_ms());
      // std::cout << "i = " << i << " : time  = " << t << " ms" << endl;
        if (i < n-1) {
            if (g) {
                for (j=l;j<n;j++)
                    v[j][i]=(a[i][j]/a[i][l])/g;
                for (j=l;j<n;j++) {
                    for (s=0.0,k=l;k<n;k++) s += a[i][k]*v[k][j];
                    for (k=l;k<n;k++) v[k][j] += s*v[k][i];
                }
            }
            for (j=l;j<n;j++) v[i][j]=v[j][i]=0.0;
        }
        v[i][i]=1.0;
        g=rv1[i];
        l=i;
    }

    // std::cout << "Accumulation of left-hand transformations" << std::endl;
    // t = (end_ms());
    // std::cout << "time  = " << t << " ms" << endl;

    QApplication::postEvent(p_viewer, new UAL::ROOT::SVDWorkerEvent(m_step++));
    mutex.lock();
    if(stopped) {
      stopped = false;
      mutex.unlock();
      return;
    }
    mutex.unlock();

    for (i=n-1; i>= 0; i--) {
      if((i/10)*10 == i) {
	QApplication::postEvent(p_viewer, new UAL::ROOT::SVDWorkerEvent(m_step++));
	mutex.lock();
	if(stopped) {
	  stopped = false;
	  mutex.unlock();
	  return;
	}
	mutex.unlock();
      }
      // t = (end_ms());
      // std::cout << "i = " << i << " : time  = " << t << " ms" << endl;
        l=i+1;
        g=w[i];
        if (i < n)
            for (j=l;j<n;j++) a[i][j]=0.0;
        if (g) {
            g=1.0/g;
            if (i != n-1) {
                for (j=l;j<n;j++) {
                    for (s=0.0,k=l;k<m;k++) s += a[k][i]*a[k][j];
                    f=(s/a[i][i])*g;
                    for (k=i;k<m;k++) a[k][j] += f*a[k][i];
                }
            }
            for (j=i;j<m;j++) a[j][i] *= g;
        } else {
            for (j=i;j<m;j++) a[j][i]=0.0;
        }
        ++a[i][i];
    }

    // std::cout << "Diagonalization of the bidiagonal form" << std::endl;
    // t = (end_ms());
    // std::cout << "time  = " << t << " ms" << endl;

    QApplication::postEvent(p_viewer, new UAL::ROOT::SVDWorkerEvent(m_step++));
    mutex.lock();
    if(stopped) {
      stopped = false;
      mutex.unlock();
      return;
    }
    mutex.unlock();

    for (k = n-1; k >= 0; k--) { // Loop over singular values
      if((k/10)*10 == k) {
	QApplication::postEvent(p_viewer, new UAL::ROOT::SVDWorkerEvent(m_step++));
	mutex.lock();
	if(stopped) {
	  stopped = false;
	  mutex.unlock();
	  return;
	}
	mutex.unlock();
      }
      // t = (end_ms());
      // std::cout << "k = " << k << " : time  = " << t << " ms" << endl;
      for (its=1;its<=30;its++) { // Loop over allowed iterations
            flag=1;
            for (l=k;l>=0;l--) { // Test for splitting
	      nm=l-1;          // Note that rv1[0] is always zero
                if (fabs(rv1[l])+anorm == anorm) {
                    flag=0;
                    break;
                }
                if (fabs(w[nm])+anorm == anorm) break;
            }
            if (flag) {
                c=0.0;
                s=1.0;
                for (i=l;i<=k;i++) {
                    f=s*rv1[i];
                    if (fabs(f)+anorm != anorm) {
                        g=w[i];
                        h=PYTHAG(f,g);
                        w[i]=h;
                        h=1.0/h;
                        c=g*h;
                        s=(-f*h);
                        for (j=0;j<m;j++) {
                            y=a[j][nm];
                            z=a[j][i];
                            a[j][nm]=y*c+z*s;
                            a[j][i]=z*c-y*s;
                        }
                    }
                }
            }
            z=w[k];
            if (l == k) { // Convergence
	      if (z < 0.0) { // Singular value is made nonnegative
                    w[k] = -z;
                    for (j=0;j<n;j++) v[j][k]=(-v[j][k]);
                }
                break;
            }

            if (its == 30) {
	      std::cout << "No convergence in 30 SVDCMP iterations" << std::endl;
	      return;
	    }

            x=w[l];
            nm=k-1;
            y=w[nm];
            g=rv1[nm];
            h=rv1[k];
            f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
            g=PYTHAG(f,1.0);
            f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;

	    // Next QR transformation.

            c=s=1.0;
            for (j=l; j <= nm; j++) {
                i=j+1;
                g=rv1[i];
                y=w[i];
                h=s*g;
                g=c*g;
                z=PYTHAG(f,h);
                rv1[j]=z;
                c=f/z;
                s=h/z;
                f=x*c+g*s;
                g=g*c-x*s;
                h=y*s;
                y=y*c;
                for (jj=0;jj<n;jj++) {
                    x=v[jj][j];
                    z=v[jj][i];
                    v[jj][j]=x*c+z*s;
                    v[jj][i]=z*c-x*s;
                }
                z=PYTHAG(f,h);
                w[j]=z;
                if (z) {
                    z=1.0/z;
                    c=f*z;
                    s=h*z;
                }
                f=(c*g)+(s*y);
                x=(c*y)-(s*g);
                for (jj=0;jj<m;jj++) {
                    y=a[jj][j];
                    z=a[jj][i];
                    a[jj][j]=y*c+z*s;
                    a[jj][i]=z*c-y*s;
                }
            }
            rv1[l]=0.0;
            rv1[k]=f;
            w[k]=x;
        }
    }
    // t = (end_ms());
    // std::cout << "end : time  = " << t << " ms" << endl;
    // QApplication::postEvent(p_viewer, new UAL::SVDWorkerEvent(m_maxStep));
    // std::cout << "Inside Worker: current step : " << m_step << std::endl;
    QApplication::postEvent(p_viewer, new UAL::ROOT::SVDWorkerEvent(m_maxStep));
}







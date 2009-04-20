#ifndef UAL_SPINK_SPIN_TRACKER_WRAPER_HH
#define UAL_SPINK_SPIN_TRACKER_WRAPER_HH

namespace SPINK {


  /** Basis class of different spin trackers */

  class SpinTrackerWriter {

  public:

    static SpinTrackerWriter* getInstance();

    void setFileName(const char* filename);

    void write(double t);


  protected:

    SpinTrackerWriter();
    
  protected:

    static SpinTrackerWriter* s_theInstance;

    std::string m_fileName;


  };

}


#endif

#ifndef UAL_OPTICS_CALCULATOR_HH
#define UAL_OPTICS_CALCULATOR_HH

#include "PAC/Beam/BeamAttributes.hh"

#include "Optics/PacVTps.h"
#include "Optics/PacChromData.h"

class Teapot;

namespace UAL {

  class OpticsCalculator {

  public:

    /** Destructor */
    ~OpticsCalculator();

    /** Returns singleton */
    static OpticsCalculator& getInstance();

    /** Sets beam attributes */
    void setBeamAttributes(const PAC::BeamAttributes& ba);

    void getOrbit(std::vector<double>& positions,
		  std::vector<PAC::Position>& orbit);

    void getTwiss(std::vector<double>& positions,
		  std::vector<PacTwissData>& twiss);

    bool use(const std::string& accName);

    /** Calculate Optics */
    bool calculate();

    /** Fits chromaticities */
    void tunefit(double tunex, double tuney,
		 std::string& b1f, std::string& b1d);

    /** Fits chromaticities */
    void chromfit(double chromx, double chromy,
		  std::string& b2f, std::string& b2d);

  public:

    /** Accelerator length */  
    double suml;

    /** Alpha */
    double alpha0;

    /** Optics */
    PacChromData* m_chrom;

  public:

    /** Beam attributes */
    PAC::BeamAttributes m_ba;

    /** Optics calculator */
    Teapot* m_teapot;

  public:

    void calculatePositions(const std::vector<int>& elems,
			    std::vector<double>& positions);
    void calculateOrbit(const std::vector<int>& elems,
			std::vector<PAC::Position>& orbitVector);
    void calculateMaps(const std::vector<int>& elems,
		       std::vector<PacVTps>& maps,
		       int order);
    void calculateTwiss(const std::vector<int>& elems,
			const std::vector<PacVTps>& maps,
			std::vector<PacTwissData>& twiss); 
    void calculateTwiss(const std::vector<int>& elems,
			const std::vector<PacVTps>& maps,
			PacTwissData& tw,
			std::vector<PacTwissData>& twiss); 

    void selectElementsByNames(const std::string& names, 
			       std::vector<int>& elemVector);   

    void writeTeapotTwissToFile(const std::string& accName,
                                const std::string& fileName, 
                                const std::string& elemNames);   

    void writeTeapotTwissToFile(const std::string& accName,
                                const std::string& fileName, 
                                const std::string& elemNames,
				PacTwissData& tw);



  private:

    static OpticsCalculator* s_theInstance;

  private:

    /** Constructor */
    OpticsCalculator();

  };

}

#endif

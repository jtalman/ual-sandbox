#ifndef UAL_SHELL_HH
#define UAL_SHELL_HH

#include "UAL/Common/Def.hh"
#include "UAL/APF/AcceleratorPropagator.hh"
#include "PAC/Beam/BeamAttributes.hh"

#include "Optics/PacVTps.h"
#include "Optics/PacTwissData.h"

#include "UAL/UI/Arguments.hh"
#include "UAL/UI/OpticsCalculator.hh"
#include "UAL/UI/BunchGenerator.hh"

namespace UAL {

  /** User Shell */

  class Shell {

  public:

    /** Constructor */
    Shell();

    /**  Defines the space of Taylor maps. 
	 @param Arg() maximum order of Taylor maps
     */
    bool setMapAttributes(const Arguments& args);

    /**  Defines the beam attributes. */
    bool setBeamAttributes(const Arguments& args);

    /** Sets bunch distribution */
    bool setBunch(const Arguments& args);

     /** Returns a bunch */
    PAC::Bunch& getBunch();      

    /** Writes ADXF file with the lattice description */
    bool writeADXF(const Arguments& args);

    /** Reads ADXF file with the lattice description */
    bool readADXF(const Arguments& args);

    /** Reads SXF file with the lattice description */
    bool readSXF(const Arguments& args);

    /** Writes SXF file with the lattice description */
    bool writeSXF(const Arguments& args);

    /** Reads APDF file with the propagator description  */
    bool readAPDF(const Arguments& args);

    /** Split elements */
    void addSplit(const UAL::Arguments& args);

    /** Selects lattice */
    bool use(const Arguments& args);

    /** Makes linear analysis */
    bool analysis(const Arguments& args);

   /** Fits tunes */
    bool tunefit(const Arguments& args);

    /** Fits chromaticities */
    bool chromfit(const Arguments& args);

    /** Make map */
    bool map(const Arguments& args);

   /** Calculate twiss */
    bool twiss(const Arguments& args);

    bool twiss(const UAL::Arguments& arguments, PacTwissData& tw);

    const std::string& getLatticeName() { return m_accName; }

    /** Calculates and returns a vector of maps */
    void getMaps(const UAL::Arguments& args, std::vector<PacVTps>& maps);

    /** Calculates and returns a vector of twiss parameters*/
    void getTwiss(const UAL::Arguments& args, 
		  std::vector<double>& positions,
		  std::vector<PacTwissData>& twiss);


    /** Runs a bunch of particles */
    bool run(const Arguments& args);

    /**  Returns a container with  the beam attributes. */
    PAC::BeamAttributes& getBeamAttributes();

    AcceleratorPropagator* getAcceleratorPropagator() { return m_ap; }
    BunchGenerator& getBunchGenerator() { return m_bunchGenerator;}

  protected:

    std::string m_accName;

    PAC::BeamAttributes m_ba;
    PAC::Bunch m_bunch;

    std::string m_apdfFile;
    UAL::AcceleratorPropagator* m_ap;

  protected:

    // Optics calculator
    // UAL::OpticsCalculator m_optics;

    BunchGenerator m_bunchGenerator;

  protected:

    bool rebuildPropagator();
    void updateBunch();

    UAL::AcceleratorNode* getLattice(const std::string& name); 
    void selectElementsByTypes(const std::string& name,
			       const std::string& types, 
			       std::vector<int>& elems);

    /*
    void calculatePositions(const std::string& name,
			    const std::vector<int>& elems,
			    std::vector<double>& positions);
    void calculateMaps(const std::vector<int>& elems,
		       std::vector<PacVTps>& maps,
		       int order);
    void calculateTwiss(const std::vector<int>& elems,
			const std::vector<PacVTps>& maps,
			std::vector<PacTwissData>& twiss);
    */




  };

}

#endif

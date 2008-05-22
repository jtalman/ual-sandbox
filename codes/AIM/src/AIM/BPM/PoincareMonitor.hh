#ifndef UAL_AIM_POINCARE_MONITOR_HH
#define UAL_AIM_POINCARE_MONITOR_HH

#include <fstream>
#include <list>

#include "UAL/Common/RCIPtr.hh"
#include "UAL/APF/PropagatorNode.hh"
#include "SMF/PacLattElement.h"
#include "PAC/Beam/Position.hh"

namespace AIM {

  /** Monitor of all bunch particles */

  class PoincareMonitor : public UAL::PropagatorNode {

  public:

    /** Constructor */
    PoincareMonitor();

    /** Copy constructor */
    PoincareMonitor(const PoincareMonitor& monitor);

    /** Destructor */
    virtual ~PoincareMonitor();

    bool isSequence() { return false; }

    /** Returns a deep copy of this object 
	(inherited from UAL::PropagatorNode) 
    */
    UAL::PropagatorNode* clone();    

    /** Set lattice elements (inherited from UAL::PropagatorNode */
    void setLatticeElements(const UAL::AcceleratorNode& lattice, 
			    int i0, int i1, 
			    const UAL::AttributeSet& beamAttributes);

    /** Returns the first node of the accelerator sector associated 
     *  with this propagator 
     * (PropagatorNode method)
     */
    virtual UAL::AcceleratorNode& getFrontAcceleratorNode();

    /** Returns the last node of the accelerator sector associated 
     * with this propagator 
     * (PropagatorNode method)
     */
    virtual UAL::AcceleratorNode& getBackAcceleratorNode();

    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);


    /** Return data */
    std::list<PAC::Bunch>& getData() { return  m_data; }   
    
    /** Clear data */
    void clear();      

    /** Returns number of turns */
    int getTurns() const { return m_data.size(); }

    /** Return a lattice name  */
    const std::string&  getLatticeName() const { return m_accName; }

    /** Return an index of the associated element */
    int getIndex() const;

    /** Returns a name of the associated element */
    const std::string&  getName() const;   

    void write(const char* fileName);


  protected:

    void init(const UAL::AcceleratorNode& lattice, int i0, int i1, 
	      const UAL::AttributeSet& beamAttributes);


  protected:

    /** accelerator name */
    std::string m_accName;

    /** element name */
    std::string m_name;

    /** element design name */
    std::string m_designName;

    /** element index */
    int m_i;

    /** element position */
    double m_s;

    /** front node */
    PacLattElement m_frontNode;

    /** back node */
    PacLattElement m_backNode;

    /** turn-by-turn data */
    std::list<PAC::Bunch> m_data;

  private:

    void init();
    void copy(const PoincareMonitor& monitor);

  };


  class PoincareMonitorRegister 
  {
    public:

    PoincareMonitorRegister(); 
  };


};

#endif

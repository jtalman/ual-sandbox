// Library       : AIM
// File          : AIM/BTF/BTFBpm.hh
// Copyright     : see Copyright file
// Author        : P.Cameron and M.Blaskiewicz
// C++ version   : N.Malitsky 

#ifndef UAL_AIM_BTF_BPM_HH
#define UAL_AIM_BTF_BPM_HH

#include "PAC/Beam/Bunch.hh"

#include "AIM/BTF/BTFBasicDevice.hh"
#include "AIM/BTF/BTFSpectrum.hh"
#include "AIM/BTF/BTFSignal.hh"

namespace AIM {

  /** BPM for Beam Transfer Function (BTF)  measurement */

  class BTFBpm : public BTFBasicDevice {

  public:

    /** Constructor */
    BTFBpm();

    /** Copy constructor */
    BTFBpm(const BTFBpm& bpm);

    /** Sets a ct bin */
    void setCtBin(double ctBin);

    /** Sets a decay parameter */
    void setTau(double tau);

    /** Sets a range of horizontal frequencies */
    void setHFreqRange(double freqLo, double freqHi, int nfreqs);

    /** Sets a range of vertical frequencies */
    void setVFreqRange(double freqLo, double freqHi, int nfreqs);

    /** Propagates a bunch of particles */
    void propagate(UAL::Probe& probe);

    /** Returns a deep copy of this node */
    UAL::PropagatorNode* clone();    

  protected:

    /** ct bin */
    double m_ctBin;

    /** decay parameter */
    double m_tau;

    /** horizontal low frequency */
    double m_hFreqLo;

    /** horizontal high frequency */
    double m_hFreqHi;

    /** number of horizontal bins */
    int m_hNFreqs;

    /** vertical low frequency */
    double m_vFreqLo;

    /** vertical high frequency */
    double m_vFreqHi;

    /** number of vertical bins */
    int m_vNFreqs;

  private:

    void init();
    void copy(const BTFBpm& bpm);

    void bunch2signal(PAC::Bunch& bunch, BTFSignal& signal);
    void signal2spectrum(BTFSignal& signal, double T, const BTFSpectrum& specIn, BTFSpectrum& specOut);

  };

}

#endif

// Program     : ETEAPOT
// File        : ETEAPOT/Integrator/ElemSlice.hh
// Copyright   : see Copyright file


#ifndef ETEAPOT_ELEM_SLICE_HH
#define ETEAPOT_ELEM_SLICE_HH

#include "Survey/PacSurveyDrift.h"
#include "Survey/PacSurveySbend.h"

namespace ETEAPOT {

  /** Container of the frame slice parameters */

  class ElemSlice
  {
  public:

    /** Constructor */
    ElemSlice();

    /** Copy constructor */
    ElemSlice(const ElemSlice& sl);

    /** Copy operator */
    ElemSlice& operator=(const ElemSlice& sl);

    /** Defines the frame slice parameters */
    void define(const PacSurvey& previous, const PacSurvey& present, const PacSurvey& next);

    /** Erases all data */
    void erase();

    /** Returns the survey object associated with this slice */
    PacSurvey& survey();

    /** Returns the survey object associated with this slice */
    const PacSurvey& survey() const;

    /** Returns phpl (phi plus), angle of this frame relative to the previous frame */
    double& phpl();

    /** Returns phpl */
    double  phpl() const;

    /** Returns cos(phpl) */
    double& cphpl();

    /** Returns cos(phpl) */
    double  cphpl() const;

    /** Returns sin(phpl) */
    double& sphpl();

    /** Returns sin(phpl) */
    double  sphpl() const;

    /** Returns tan(phpl) */
    double& tphpl();

    /** Returns tan(phpl) */
    double  tphpl() const ;

    /** Returns a distance between this and next slices */
    double& rlipl();

    /** Returns a distance between this and next slices */    
    double  rlipl() const;

    /** Returns the x coordinate of the next slice */
    double& scrx();

    /** Returns the x coordinate of the next slice */
    double  scrx() const;

    /** Returns the s coordinate of the next slice */
    double& scrs();

    /** Returns the s coordinate of the next slice */    
    double  scrs() const;

    /** Returns scrs + scrx*tphpl */
    double& spxt();

    /** Returns scrs + scrx*tphpl */
    double  spxt() const;

  protected:

    /** Survey associated with this slice */
    PacSurvey _survey;

    // Some precalculations used in tracking

    /** phpl (phi plus), angle of this frame relative to the previous frame */
    double _phpl;

    /** cos(phpl) */
    double _cphpl;

    /** sin(phpl) */
    double _sphpl; 

    /** tan(phpl) */
    double _tphpl; 

    /** distance between this and next slices */
    double _rlipl; 

    /** x coordinate of the next slice */
    double _scrx;

    /** s coordinate of the next slice */
    double _scrs; 

    /** scrs + scrx*tphpl */
    double _spxt;

  private:

    void initialize();
    void initialize(const ElemSlice& sl);

  };

}

#endif

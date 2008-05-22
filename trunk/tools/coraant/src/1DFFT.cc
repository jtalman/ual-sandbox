
#include <iostream>
#include "1DFFT.hh"

using namespace std;

CORAANT::OneDFFT::OneDFFT(int s, int ps)
{
  size=s;
  psize = (ps) ? ps : size;
  //cout<<"THe size of the arrays is "<<size<<endl;
  //  cout<<"THe power spectrum size is "<<psize<<endl;
   if (psize!=0){
    power_spectrum=new Double_t [psize];
    if (!power_spectrum){
      cerr<<"OneDFFT: Cannot allocate enough memory"<<endl;
      size=psize=0;
    }
  }
  else power_spectrum=NULL;

  totalPower=0;
  minPowerBin=0;
  maxPowerBin=0;
}


CORAANT::OneDFFT::~OneDFFT()
{
  if (power_spectrum) delete [] power_spectrum;
}


CORAANT::real1DFFT::real1DFFT(int s):OneDFFT(s,s/2+1)
{
 
  if (size){
    in=(fftw_real *)fftw_malloc(sizeof(fftw_real)*size);
    out=(fftw_real *)fftw_malloc(sizeof(fftw_real)*size);
  }
  deleteIn=deleteOut=true;
  if(!in || !out){
    cerr<<"real1DFFT: Cannot allocate enough memory"<<endl;
    if (in) fftw_free(in);
    if (out) fftw_free(out);
    if (power_spectrum) delete [] power_spectrum;
    size=psize=0;
  }
  createPlan();

}

CORAANT::real1DFFT::real1DFFT(int s, fftw_real *inp, fftw_real *outp):OneDFFT(s,s/2+1)
{
  in=inp;
  deleteIn=false;
  if(!outp){
    deleteOut=true;
    out=(fftw_real *)fftw_malloc(sizeof(fftw_real)*size);
    if(!out){
      cerr<<"real1DFFT: Cannot allocate enough memory"<<endl;
      size=psize=0;
      if (power_spectrum) delete [] power_spectrum;
    }
  }else deleteOut=false;
  createPlan();

}

CORAANT::real1DFFT::~real1DFFT()
{
  destroyPlan();
  if(in && deleteIn) fftw_free(in);
  if(out && deleteOut) fftw_free(out);
 
}

void CORAANT::real1DFFT::createPlan()
{
  
#if FFTWVersion == 2
    fwdPlan = rfftw_create_plan(size, FFTW_REAL_TO_COMPLEX, PLANTYPE);
    invPlan = rfftw_create_plan(size, FFTW_COMPLEX_TO_REAL, PLANTYPE);
#else
    fwdPlan = fftw_plan_r2r_1d(size, in, out, FFTW_R2HC, PLANTYPE);
    invPlan = fftw_plan_r2r_1d(size, out, in, FFTW_HC2R, PLANTYPE);
#endif

}

void CORAANT::real1DFFT::destroyPlan()
{
  fftw_destroy_plan(fwdPlan);
  fftw_destroy_plan(invPlan);
}

void CORAANT::real1DFFT::switchData(int s, fftw_real *inp)
{
  int i;

#if FFTWVersion == 3
  destroyPlan();
#endif

  if (size==0 || size!=s){ //reconstruct the plan  && reallocate arrays
    size=s;
    psize=s/2+1;
#if FFTWVersion == 2
    destroyPlan();
#endif
    if(s>size && !deleteOut) cerr<<"real1DFFT::switchData the output array is too small to fit the FFT"<<endl;
    if(out && deleteOut) {
      fftw_free(out);
      out=(fftw_real *)fftw_malloc(sizeof(fftw_real)*size);
    }
    if (power_spectrum) delete [] power_spectrum;
    power_spectrum=new Double_t [psize];
    if(!out || !power_spectrum){
       cerr<<"real1DFFT::switchData Cannot allocate enough memory"<<endl;
       if (out) fftw_free(out);
       if (power_spectrum) delete [] power_spectrum;
       size=psize=0;
    }
#if FFTWVersion == 2
    createPlan();
#endif
  } 
  if(in && deleteIn) {fftw_free(in); deleteIn=false;} //delete old array if necessary
  in=inp;
  for(i=0;i<size;i++) out[i]=0.0;  //zero the old FFT and power spectrum
  for(i=0;i<psize;i++) power_spectrum[i]=0.0;

#if FFTWVersion == 3
    createPlan();
#endif

  totalPower=0;
  minPowerBin=0;
  maxPowerBin=0;
}


int CORAANT::real1DFFT::compPowerSpec()
{
  double max,min;
  int k;
  totalPower = 0.0;

  //  cout<<"in compPowerSpec "<<endl;
  power_spectrum[0] = out[0]*out[0];  /* DC component */
  max= min=power_spectrum[0];
  

  for (k = 1; k < psize; ++k){
    //cout<<"k = "<<k<<" out of "<<psize<<" and size-k - "<<(size-k)<<"out of"<< size<<endl;
    power_spectrum[k] = out[k]*out[k] + out[size-k]*out[size-k];
    if (power_spectrum[k]>max){
      max=power_spectrum[k];
      maxPowerBin=k;
    }
    if (power_spectrum[k]<min){
      min=power_spectrum[k];
      minPowerBin=k;
    }
    totalPower += power_spectrum[k];
  }
  //cout<<"out of k loop"<<endl;
  if (size % 2 == 0){ /* N is even */
    //cout<<"even power spectrum"<<endl;
    power_spectrum[size/2] = out[size/2]*out[size/2];  /* Nyquist freq. */
    if (power_spectrum[size/2]>max){
      max=power_spectrum[size/2];      
      maxPowerBin=size/2;
    }
    if (power_spectrum[size/2]<min){
      min=power_spectrum[size/2];
      minPowerBin=k;
    }
    totalPower += power_spectrum[size/2];
  }

  //cout<<"About to return"<<endl;
  return maxPowerBin;
}

  
CORAANT::complex1DFFT::complex1DFFT(int s):OneDFFT(s,s)
{
 
  if (size){
    in=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*psize);
    out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*psize);
  }
  deleteIn=deleteOut=true;
  if(!in || !out){
    cerr<<"complex1DFFT: Cannot allocate enough memory"<<endl;
    if (in) fftw_free(in);
    if (out) fftw_free(out);
    if (power_spectrum) delete [] power_spectrum;
    size=psize=0;
  }
  createPlan();

}

CORAANT::complex1DFFT::complex1DFFT(int s, fftw_complex *inp, fftw_complex *outp):OneDFFT(s,s/2+1)
{
  in=inp;
  deleteIn=false;
  if(!outp){
    deleteOut=true;
    out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*psize);
    if(!out){
      cerr<<"complex1DFFT: Cannot allocate enough memory"<<endl;
      size=psize=0;
      if (power_spectrum) delete [] power_spectrum;
    }
  }else deleteOut=false;
  createPlan();

}

CORAANT::complex1DFFT::~complex1DFFT()
{
  if(in && deleteIn) fftw_free(in);
  if(out && deleteOut) fftw_free(out);
  destroyPlan();
}

void CORAANT::complex1DFFT::createPlan()
{
#if FFTWVersion == 2
    fwdPlan = fftw_create_plan(size, FFTW_FORWARD, PLANTYPE);
    invPlan = fftw_create_plan(size, FFTW_BACKWARD, PLANTYPE);
#else
    fwdPlan = fftw_plan_dft_1d(size, in, out, FFTW_FORWARD, PLANTYPE);
    invPlan = fftw_plan_dft_1d(size, out, in, FFTW_BACKWARD, PLANTYPE);
#endif
}

void CORAANT::complex1DFFT::destroyPlan()
{
  fftw_destroy_plan(fwdPlan);
  fftw_destroy_plan(invPlan);
}

void CORAANT::complex1DFFT::switchData(int s, fftw_complex *inp)
{
  int i;

#if FFTWVersion == 3
    destroyPlan();
#endif

  if (size==0 || size!=s){ //reconstruct the plan  && reallocate arrays
    size=s;
    psize=s;
#if FFTWVersion == 2
    destroyPlan();
#endif   
    if(s>size && !deleteOut) cerr<<"real1DFFT::switchData the output array is too small to fit the FFT"<<endl;
    if(out && deleteOut) {
      fftw_free(out);
      out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*size);
    }
    if (power_spectrum) delete [] power_spectrum;
    power_spectrum=new Double_t [psize];
    if(!out || !power_spectrum){
       cerr<<"complex1DFFT::switchData Cannot allocate enough memory"<<endl;
       if (out) fftw_free(out);
       if (power_spectrum) delete [] power_spectrum;
       size=psize=0;
    }
#if FFTWVersion == 2
    createPlan();
#endif
  } 


  if(in && deleteIn) {fftw_free(in); deleteIn=false;}  //delete old array if necessary
  in=inp;
  for(i=0;i<size;i++){c_re(out[i])=0.0;c_im(out[i])=0;}  //zero the old FFT and power spectrum
  for(i=0;i<psize;i++) power_spectrum[i]=0.0;

#if FFTWVersion == 3
    createPlan();
#endif

  totalPower=0;
  minPowerBin=0;
  maxPowerBin=0;
}


int CORAANT::complex1DFFT::compPowerSpec()
{
  double max,min;
  int k;
  totalPower = 0.0;
  
  power_spectrum[0] = c_re(out[0])*c_re(out[0])+c_im(out[0])*c_im(out[0]);  /* DC component */
  max= min=power_spectrum[0];
  

  for (k = 1; k < psize; ++k){
    power_spectrum[k] = c_re(out[k])*c_re(out[k])+c_im(out[k])*c_im(out[k]);
    if (power_spectrum[k]>max){
      max=power_spectrum[k];
      maxPowerBin=k;
    }
    if (power_spectrum[k]<min){
      min=power_spectrum[k];
      minPowerBin=k;
    }
    totalPower += power_spectrum[k];
  }

  return maxPowerBin;
}
  

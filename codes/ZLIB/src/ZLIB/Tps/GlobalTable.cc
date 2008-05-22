// Library     : ZLIB
// File        : ZLIB/Tps/GlobalTable.cc
// Copyright   : see Copyright file
// Author      : Yiton Yan
// C++ version : Nikolay Malitsky

#include "ZLIB/Tps/GlobalTable.h"

ZLIB::GlobalTable::GlobalTable(int dim, int o)
  : counter(0)
{
  if(dim < 1) 
  {
     cerr << "Error: ZLIB::GlobalTable::GlobalTable(dim, order) : dim < 1\n"; 
     assert(dim > 0);
  } 
  
  if(o < 1) 
  {
     cerr << "Error: ZLIB::GlobalTable::GlobalTable(dim, order) : order < 1\n"; 
     assert(order > 0);
  }   

  dimension = dim;
  order     = o;

  defineTpsData(dim, o);
  defineVTpsData(dim, o);
            
}

ZLIB::GlobalTable::~GlobalTable()
{
  eraseTpsData();
  eraseVTpsData();
}

// Tps data

void ZLIB::GlobalTable::defineTpsData(int dim, int o)
{ 
  int i, j;
  int DA_DIM = dim;
  int DA_ORDER = o;             
     
//--------------------------------------------------------------
// nmo [DA_ORDER+2]
// ZLIB::: subroutine nmonkp(nv,no,nmo,nmob,nmm,nikpm,nikpm2,nvms)
//--------------------------------------------------------------

 nmo = new int[DA_ORDER+2];
 if(!nmo)
 {
     cerr << "Error: ZLIB::GlobalTable::ZLIB::GlobalTable(dim, maxOrder) : allocation failure in nmo\n"; 
     assert(nmo);
 }     
 nmo[0] = 1;

 for(i=1; i <= DA_ORDER+1; i++)
        nmo[i] = (nmo[i-1]*(DA_DIM+i))/i;

//--------------------------------------------------------------
// nmov[DA_ORDER+3][DA_ORDER+2][DA_DIM] 
// ZLIB::: subroutine nmonv(nv,no,nmov)
//--------------------------------------------------------------

 nmov = new int**[DA_ORDER+3];
 for( i=0; i <= DA_ORDER+2;i++)
 {
   nmov[i] = new int*[DA_ORDER+2];
   for( j=0; j <= DA_ORDER+1; j++)
   {
       nmov[i][j] = new int[DA_DIM];
       for(int k=0; k < DA_DIM; k++)
           nmov[i][j][k] = 0;
   }
 }

 if(!nmov[DA_ORDER+2][DA_ORDER+1])
 {
     cerr << "Error: ZLIB::GlobalTable::ZLIB::GlobalTable(dim, maxOrder) : allocation failure in nmov\n"; 
     assert(nmov[DA_ORDER+2][DA_ORDER+1]);
 } 

 int noa, iv; 
 for(iv=0; iv < DA_DIM; iv++)
  for(noa=0; noa <= DA_ORDER+1;noa++)
                      nmov[noa+1][noa][iv]=0;
 for(iv=0; iv < DA_DIM; iv++)
            nmov[0][0][iv] = 1;
 int iox;
 for(noa=0;noa <= DA_ORDER+1;noa++)
   for(iox=0; iox <= noa;iox++)
         nmov[iox][noa][DA_DIM-1]=noa+1-iox;
 for(iv=DA_DIM-2; iv >= 1; iv--)
    for(noa=0;noa <= DA_ORDER+1;noa++)
       for(iox=noa; iox >= 0; iox--)
           nmov[iox][noa][iv] = nmov[iox+1][noa][iv]+nmov[0][noa-iox][iv+1];

//---------------------------------------------------------------
// jv[DA_DIM+1][nmo[DA_ORDER]+1]
// ZLIB::: subroutine zpokjv(nv,no,nm,nmo,nmov,jv)
//--------------------------------------------------------------- 

 jv = new int*[DA_DIM+1];
 for(iv=0; iv <= DA_DIM; iv++)
 {
     jv[iv] = new int[nmo[DA_ORDER]+1];
     for(int k=0; k <= nmo[DA_ORDER]; k++)
         jv[iv][k] = 0;
 }
 if(!jv[DA_DIM])
 {
     cerr << "ZLIB::GlobalTable::ZLIB::GlobalTable(dim, maxOrder) : allocation failure in jv \n"; 
     assert(jv[DA_DIM]);
 }   

 for(iv=0; iv <= DA_DIM;iv++)
 {
         jv[iv][0] = 0;
         jv[iv][1] = 0;
 }

 for(j=2; j <= nmo[DA_ORDER];j++)
 {
     int io;
     for(io=0; io <= DA_ORDER;io++)
               if(j <= nmo[io]) break;
     jv[0][j] = io;
     int j1 = j-nmo[io-1];
     noa = io;
     for(iv=1; iv < DA_DIM; iv++)
     {
        for(iox=noa; iox >= 0; iox--)
             if(j1 <= nmov[iox][noa][iv]) break;
        jv[iv][j] = iox;
        j1 -= nmov[iox+1][noa][iv];
        noa -= iox;
      }
      jv[DA_DIM][j] = noa;
  }

//--------------------------------------------------------------
// nkpm
// js1[DA_DIM+1]
// ZLIB::: subroutine zpnkpm(nv,nm,jv,nkpm,nkpm2)
//--------------------------------------------------------------

 js1 = new int[DA_DIM+1];
 int k;
 for(k=0; k <= DA_DIM; k++) 
     js1[k] = 0;
 if(!js1)
 {
     cerr << "Error: ZLIB::GlobalTable::ZLIB::GlobalTable(dim, maxOrder) : allocation failure in js1\n"; 
     assert(js1);
 } 

 nkpm = 1;
 for(j=2; j <= nmo[DA_ORDER]; j++)
 {
    for(iv=1; iv <= DA_DIM; iv++)
                      js1[iv]=jv[iv][j];
    int nkpj = js1[1]+1;
    for(iv=2; iv <= DA_DIM; iv++)
                      nkpj *= (js1[iv]+1);
    nkpm+=nkpj;
  }

//--------------------------------------------------------------
// ikb[DA_ORDER+1][nmo[DA_ORDER]+1]
// ikp[DA_ORDER+1][nmo[DA_ORDER]+1]
// kp [nkpm+1]
// lp [nkpm+1]
// ZLIB::: subroutine muprp(nv,no,nm,ikb,ikp,jv,i)
//--------------------------------------------------------------

 ikb = new int*[DA_ORDER+1];
 ikp = new int*[DA_ORDER+1];
 for(i=0; i<= DA_ORDER; i++)
 {
      ikb[i] = new int[nmo[DA_ORDER]+1];
      ikp[i] = new int[nmo[DA_ORDER]+1];
      for(k=0; k <= nmo[DA_ORDER]; k++)
      {
         ikb[i][k] = 0;
         ikp[i][k] = 0;
      }         
 }

 if(!ikp[DA_ORDER])
 {
    cerr << "Error: ZLIB::GlobalTable::ZLIB::GlobalTable(dim, maxOrder) : allocation failure in ikb,ikp \n"; 
    assert(ikp[DA_ORDER]);
 } 

 kp  = new int [nkpm+1];
 lp  = new int [nkpm+1];
 for(k=0; k <= nkpm; k++)
 {
   kp[k] = 0;
   lp[k] = 0;
 }
 if(!lp)
 {
    cerr << "Error: ZLIB::GlobalTable::ZLIB::GlobalTable(dim, maxOrder) : allocation failure in kp,lp \n"; 
    assert(lp);
 } 

 i = 0;
 for(int io=2; io <= DA_ORDER; io++)
 {
    for(j=nmo[io-1]+1; j <=nmo[io]; j++)
    {
      int iou = 0;
	i++;
        ikp[0][j] = i;
	ikb[0][j] = i;
        kp[i]     = 1;
        lp[i]     = j;
        for(iou=1; iou < io; iou++)
        {
            for(int ju=nmo[iou-1]+1; ju <= nmo[iou];ju++)
            {
               int itis = 1; // TRUE
               for(iv=1; iv <= DA_DIM;iv++)
                   if(jv[iv][ju] > jv[iv][j]) itis = 0; // FALSE
               if(itis) 
               {
                   i++;
                   kp[i] = ju;
                   for(iv=1; iv <= DA_DIM; iv++)
                        js1[iv] = jv[iv][j]-jv[iv][ju];
                   lp[i] = jpek(js1);
                }
             }
             ikp[iou][j] = i;
             ikb[iou][j] = ikp[iou-1][j]+1;
         }
//  iou=io
         i++;
         ikp[io][j] = i;
         ikb[io][j] = ikp[io-1][j]+1;
         kp     [i] = j;
         lp     [i] = 1;
     }
  }

 // NM. Transfer FORTRAN indexing to C++

 for(k=0; k <= nkpm; k++)
 {
   kp[k] -= 1;
   lp[k] -= 1;
 }

//---------------------------------------------------------
// jd [DA_DIM+1][nmo[DA_ORDER]+1]
// ZLIB::: subroutine drvprp(nv, nm, njd, jd, jv)
//---------------------------------------------------------
 jd = new int*[DA_DIM+1];
 for(i=0; i <= DA_DIM; i++)
 {
   jd[i] = new int[nmo[DA_ORDER]+1];
   for(k=0; k <= nmo[DA_ORDER]; k++)
       jd[i][k] = 0;
 }

 if(!jd[DA_DIM])
 {
     cerr << "Error: ZLIB::GlobalTable::ZLIB::GlobalTable(dim, maxOrder) : allocation failure in jd \n"; 
     assert(jd[DA_DIM]);
 }   

 for(j=1; j <= nmo[DA_ORDER]; j++)
 {
    for(i=1; i <= DA_DIM; i++)
        js1[i] = jv[i][j];
    js1[1] += 1;
    jd[1][j] = jpek(js1);
    for(iv=2; iv <= DA_DIM; iv++)
    {
       js1[iv-1] -= 1;
       js1[iv]   += 1;
       jd[iv][j] = jpek(js1);
     }
 } 
 
 // NM. Transfer FORTRAN indexing to C++

 for(i=0; i <= DA_DIM; i++)
   for(j=1; j <= nmo[DA_ORDER]; j++)
     jd[i][j] -= 1;

}

int ZLIB::GlobalTable::jpek(int* jps)
{
   int DA_DIM = dimension;

   int iv;
   if(DA_DIM == 1) return(jps[1]+1);
   int io = 0;
   for(iv = 1; iv <= DA_DIM; iv++)
                           io += jps[iv];
   if(io <= 0) return(1);
   int jppek = nmo[io-1];
   int noa = io;
   for(iv = 1; iv <= DA_DIM-2; iv++)
   {
     jppek += nmov[jps[iv]+1][noa][iv];
     noa   -= jps[iv];
   }
   jppek += nmov[jps[DA_DIM-1]][noa][DA_DIM-1];
   return(jppek);
}

void ZLIB::GlobalTable::eraseTpsData()
{
  int DA_DIM   = dimension;
  int DA_ORDER = order;  

  if(nmo) delete [] nmo;

  if(jv){
    for(int i=0; i < DA_DIM+1; i++)
      if(jv[i]) delete [] jv[i];

    delete [] jv;
  }

// Members for Multiplication

  if(kp) delete [] kp;

  if(lp) delete [] lp;

  if(ikp){
    for(int i=0; i < DA_ORDER+1; i++)
      if(ikp[i]) delete [] ikp[i];

    delete [] ikp;
  } 
 
  if(ikb){
    for(int i=0; i < DA_ORDER+1; i++)
      if(ikb[i]) delete [] ikb[i];

    delete [] ikb;
  }  

// Members for Derivative

  if(jd){
    for(int i=0; i < DA_DIM+1; i++)
      if(jd[i]) delete [] jd[i];

    delete [] jd;
  } 

// Auxiliary Static Members 
// nmov --> jv --> js1 --> jpek() --> Static Members for Mult.
//          jv --> js1 --> jpec() --> Static Members for Deriv.
 
  if(nmov) {
    int i;
    for(i=0; i < DA_ORDER+3; i++)
      for(int j=0; j < DA_ORDER+2; j++)
	if(nmov[i][j]) delete [] nmov[i][j];

    for(i=0; i < DA_ORDER+3; i++)
      if(nmov[i]) delete [] nmov[i];

    delete [] nmov;
  }

  if(js1) delete [] js1;

}

void ZLIB::GlobalTable::defineVTpsData(int dim, int o)
{
  int i = 0, j = 0, k = 0;

  int DA_DIM   = dim;
  int DA_ORDER = o;

//--------------------------------------------------------------
// mp [DA_ORDER+1][DA_ORDER+1][DA_DIM+1]
// ZLIB::: subroutine zpmp(nv,no,mp)
//--------------------------------------------------------------

 mp = new int**[DA_ORDER+1];
 for( i=0; i <= DA_ORDER;i++)
 {
   mp[i] = new int*[DA_ORDER+1];
   for( j=0; j <= DA_ORDER; j++)
   {
       mp[i][j] = new int[DA_DIM+1];
       for( k=0; k <= DA_DIM; k++)
          mp[i][j][k] = 0;
   }
 }

 if(!mp[DA_ORDER][DA_ORDER]) 
 {
    cerr << "Error: ZLIB::GlobalTable::ZLIB::GlobalTable(dim, order) : allocation failure for mp \n";
    assert(mp[DA_ORDER][DA_ORDER]);
 }

 for(i=0; i <= DA_ORDER; i++)
   for(j=0; j <= DA_ORDER; j++)
                  mp[i][j][1]=i;
 for(i=1; i <= DA_ORDER; i++)
    for(j=1; j <= DA_ORDER; j++)
         mp[i][j][2] = mp[i-1][j][2]+j-i+2;
 for(k=3; k <= DA_DIM; k++)
   for(j=1; j <= DA_ORDER; j++)
   {
          for(i=1; i <= j; i++)
            mp[i][j][k] = mp[j-i+1][j-i+1][k-1] + 1;
          for(i=1; i<=j; i++)
            mp[i][j][k]+=mp[i-1][j][k];
   } 


//--------------------------------------------------------------
// jpc  [nmo[DA_ORDER]-DA_DIM]
// ivpc [nmo[DA_ORDER]-DA_DIM]
// ivppc[nmo[DA_ORDER]-DA_DIM]
// ZLIB::: subroutine pntcct(nv,no,nm,nvs,nos,mp)
//-------------------------------------------------------------- 

 int axsize = nmo[DA_ORDER]-DA_DIM ;
 jpc  = new int[axsize];
 ivpc = new int[axsize];
 ivppc= new int[axsize];
 for(i=0; i < axsize; i++)
 {
   jpc[i]   = 0;
   ivpc[i]  = 0;
   ivppc[i] = 0;
 }

 js2  = new int[DA_DIM+1];
 
 for(i=0 ; i <= DA_DIM; i++)
   js2[i] = 0;

 if(!js2) 
 {
    cerr << "Error: ZLIB::GlobalTable::ZLIB::GlobalTable(int dim, int order) : allocation failure for(jpc-js2)\n";
    assert(js2);
 }

//--------------------------------------------------------------
// nmvo [DA_DIM + 1][DA_ORDER + 1]
// ivp  [nmo[DA_ORDER] + 1]
// jpp  [nmo[DA_ORDER] + 1]
// ZLIB::: subroutine prptrk(nv,no,nm,nmo,nmvo,ivp,jpp)
//--------------------------------------------------------------

 nmvo = new int*[DA_DIM+1];
 for(i=0; i <= DA_DIM; i++)
 {
    nmvo[i] = new int[DA_ORDER+1];
    for(k=0; k <= DA_ORDER; k++)
       nmvo[i][k] = 0;
 }

 axsize = nmo[DA_ORDER] + 1;

 ivp  = new int[axsize];
 jpp  = new int[axsize];

 for(i=0; i < axsize; i++)
 {
   ivp[i]  = 0;
   jpp[i]  = 0;
 }

 if(!jpp) 
 {
    cerr << "Error: ZLIB::GlobalTable::ZLIB::GlobalTable(int dim, int order) : allocation failure for(nmvo-jpp)\n";
    assert(jpp);
 } 
 int iv, io;
 for(iv=1; iv <= DA_DIM; iv++) 
    nmvo[iv][1] = 1;
 for(io=2; io <= DA_ORDER; io++)
 {
    int iom1 = io - 1;
    for(iv=1; iv <= DA_DIM; iv++)
    {
       nmvo[iv][io] = nmvo[iv][iom1];
       for(i=iv+1; i <= DA_DIM; i++)
           nmvo[iv][io] += nmvo[i][iom1];
    }
 }

// io = 1

 for(int jj=1; jj <= DA_DIM; jj++)
 {
    j = jj +1;
    ivp[j] = jj;
    jpp[j] = 1;
 }

// io = 2, DA_ORDER

  for(io=2; io <= DA_ORDER; io++)
  {
     int iom1 = io -1;
     int iom2 = io -2;
     for(iv=1; iv <= DA_DIM; iv++)
     {
        int nmvovo = nmvo[iv][io];
        int ivm1   = iv - 1;
        for(int jvo=1; jvo <= nmvovo; jvo++)
        {
            j += 1;
            ivp[j] = iv;
            jpp[j] = nmo[iom2] + jvo;
            for(i=1; i <= ivm1; i++)
               jpp[j] += nmvo[i][iom1];
        }
     }
  }

}

void ZLIB::GlobalTable::pntcct(int no)
{
 int i;
 int j;
 int jj = 0;
 int io;

 int DA_DIM = dimension;

 for(int jjj = 2; jjj <= nmo[no]; jjj++)
 {
   ztpapek(jjj, no);
   io=js2[1];
   for(i=2; i <= DA_DIM; i++) io += js2[i];
   if(io > 1)
   {     
      jj += 1;
      j = jpek(js2); 
      jpc[jj] = j;
      for(int iv=1; iv <= DA_DIM; iv++)
      {
         if(js2[iv] > 1)
         {
           ivpc [jj] = iv;
           ivppc[jj] = iv;
           break;
         }
         if(js2[iv] == 1)
         {
           ivpc[jj] = iv;
           for(int ii=iv+1; ii <= DA_DIM; ii++)
           {
               if(js2[ii] > 0)
               {
                  ivppc[jj] = ii;
                  break;
               }
               if(ii > DA_DIM)
               {
                  cerr << "Error: ZLIB::GlobalTable::pntcct :";
                  assert(ii <= DA_DIM);
               }
           }
           break;
         }
       }
    }
  }
  return;
}

void ZLIB::GlobalTable::ztpapek(int jtpa, int no)
{
   int i;
   int mona = jtpa;
   int noa = no;

   int DA_DIM = dimension;

   for(int iv = DA_DIM; iv >= 2; iv--)
     for(i=noa; i >= 0; i--)
     {
         if(mona > mp[i][noa][iv])
         {
            js2[iv] = i;
            mona -= mp[i][noa][iv];
            noa -= i;
            break;
          }
      }
   for(i=noa; i >= 0; i--)
      if(mona > mp[i][noa][1])
      {
        js2[1] = i;
        break;
      }
   return;
}

void ZLIB::GlobalTable::eraseVTpsData()
{
  int DA_DIM   = dimension;
  int DA_ORDER = order;  

  // Members for Multiplication

  if(jpc) delete [] jpc;
  if(ivpc) delete [] ivpc;
  if(ivppc) delete [] ivppc;

  // Members for Tracking

  if(ivp) delete [] ivp;
  if(jpp) delete [] jpp;

  // ZLIB:: auxiliary Static Members 
  // mp --> js, ztpapek()--> jpek() --> pntcct() --> Static Members for Mult.
  // nmvo --> Static Members for Tracking

  if(mp){
    int i;
    for(i=0; i <= DA_ORDER;i++)
      for(int j=0; j <= DA_ORDER; j++)
	delete [] mp[i][j];

    for(i=0; i <= DA_ORDER;i++)
      delete [] mp[i];

    delete [] mp;
  }

  if(nmvo){
    for(int i=0; i <= DA_DIM; i++) 
      delete [] nmvo[i];
    delete [] nmvo;
  }

  if(js2) delete [] js2;

}

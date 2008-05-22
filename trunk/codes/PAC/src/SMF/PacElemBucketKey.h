// Library     : PAC
// File        : SMF/PacElemBucketKey.h
// Copyright   : see Copyright file
// Author      : Nikolay Malitsky

#ifndef PAC_ELEM_BUCKET_KEY_H
#define PAC_ELEM_BUCKET_KEY_H

#include "Templates/PacRCIPtr.h"

class PacElemAttribKey;

class PacElemBucketKey
{

public:

// Constructors & copy operator

  PacElemBucketKey() : _ptr(new PacElemBucketKey::Data()) {}
  PacElemBucketKey(const char* name, int key, int size, int order, const char* attKeys);
  PacElemBucketKey(const PacElemBucketKey& key) : _ptr(key._ptr) {}

  PacElemBucketKey& operator  = (const PacElemBucketKey& key) { checkName(); _ptr = key._ptr; return *this;}

// Key & Count

   const int& key() const { return _ptr->_key; }
   int count() const { return _ptr.count(); }

// Data

   const std::string& name() const { return _ptr->_name; }

   int size() const { return _ptr->_size; }
   const PacElemAttribKey& operator[](int index) const ;

   int order() const { return _ptr->_order; }

// Interface to collection items

   PacElemBucketKey* operator()(int key) const;

protected:

   class Data
   {
   public:

     Data() : _key(0), _size(0), _attKeys(0), _order(0) {}
     ~Data();

     int _key;
     std::string _name;

     int _size;
     PacElemAttribKey* _attKeys;

     int _order;
 
   };

   typedef PacRCIPtr<Data> smart_pointer;
   smart_pointer _ptr;

private:

  void checkName();
  void create(const char* names);

};

struct PacKeyOfElemBucketKey
{
  const int& operator()(const PacElemBucketKey& x) const { return x.key();} 

  // it is prohibited to prevent the insertion of new PacElemBucketKey
  void operator()(PacElemBucketKey& x, int key) const; 

  // it is prohibited to prevent the erasion of PacElemBucketKey
  int count(const PacElemBucketKey& x) const ;
};

class PacElemAttribKey
{

public:

// Constructors & copy operator

   PacElemAttribKey() : _index(0), _order(0) {}
   PacElemAttribKey(const PacElemAttribKey& key) { define(key); }

   PacElemAttribKey& operator = (const PacElemAttribKey& key) { define(key); return *this;}

// Data

   const std::string& name() const { return _name; }

   int index() const { return _index + _order*bucketKey().size(); }
   int order() const { return _order; }

   const PacElemBucketKey& bucketKey() const { return _bucketKey; }

   PacElemAttribKey operator()(int order) const ;

// Friend

   friend class PacElemBucketKey;

protected:

   PacElemBucketKey _bucketKey;
 
   std::string _name;

   int _index;   
   int _order;

private:

   void checkName();
   void define(const PacElemAttribKey& key);
   void define(const PacElemBucketKey& bucketKey, const std::string& name, int index);

};

#define PAC_LENGTH   1
#define PAC_LENGTH_L 0
#define PAC_LENGTH_SIZE 1

extern PacElemBucketKey pacLength;      // pacLength("Length", PAC_LENGTH, PAC_LENGTH_SIZE, 0, "L");
#define PAC_L pacLength[PAC_LENGTH_L]

#define PAC_BEND 2
#define PAC_BEND_ANGLE 0
#define PAC_BEND_FINT  1
#define PAC_BEND_SIZE  2

extern PacElemBucketKey pacBend;        // pacBend("Bend", PAC_BEND, PAC_BEND_SIZE, 0, "ANGLE FINT");
#define PAC_ANGLE pacBend[PAC_BEND_ANGLE]
#define PAC_FINT  pacBend[PAC_BEND_FINT]

#define PAC_MULTIPOLE 3
#define PAC_MULTIPOLE_KL 0
#define PAC_MULTIPOLE_KTL 1
#define PAC_MULTIPOLE_SIZE 2

extern PacElemBucketKey pacMultipole;  // pacNmultipole("Multipole", PAC_MULTIPOLE, PAC_MULTIPOLE_SIZE, 1, "KL KTL");
#define PAC_KL  pacMultipole[PAC_MULTIPOLE_KL]
#define PAC_KTL pacMultipole[PAC_MULTIPOLE_KTL]

#define PAC_OFFSET 4
#define PAC_OFFSET_DX 0
#define PAC_OFFSET_DY 1
#define PAC_OFFSET_DS 2
#define PAC_OFFSET_SIZE 3

extern PacElemBucketKey pacOffset;      // pacOffset("Offset", PAC_OFFSET, PAC_OFFSET_SIZE, 0, "DX DY DS");
#define PAC_DX pacOffset[PAC_OFFSET_DX]
#define PAC_DY pacOffset[PAC_OFFSET_DY]
#define PAC_DS pacOffset[PAC_OFFSET_DS]

#define PAC_ROTATION 5
#define PAC_ROTATION_DPHI   0
#define PAC_ROTATION_DTHETA 1
#define PAC_ROTATION_TILT   2
#define PAC_ROTATION_SIZE   3

extern PacElemBucketKey pacRotation;    // pacRotation("Rotation", PAC_ROTATION, PAC_ROTATION_SIZE, 0, "DPHI DTHETA TILT");
#define PAC_DPHI   pacRotation[PAC_ROTATION_DPHI]
#define PAC_DTHETA pacRotation[PAC_ROTATION_DTHETA]
#define PAC_TILT   pacRotation[PAC_ROTATION_TILT]

#define PAC_APERTURE 6
#define PAC_APERTURE_SHAPE 0
#define PAC_APERTURE_XSIZE 1
#define PAC_APERTURE_YSIZE 2
#define PAC_APERTURE_SIZE  3

extern PacElemBucketKey pacAperture;    // pacAperture("Aperture", PAC_APERTURE, PAC_APERTURE_SIZE, 0, "SHAPE XSIZE YSIZE");
#define PAC_SHAPE   pacAperture[PAC_APERTURE_SHAPE]
#define PAC_XSIZE   pacAperture[PAC_APERTURE_XSIZE]
#define PAC_YSIZE   pacAperture[PAC_APERTURE_YSIZE]

#define PAC_COMPLEXITY 7
#define PAC_COMPLEXITY_N    0
#define PAC_COMPLEXITY_SIZE 1

extern PacElemBucketKey pacComplexity;  // pacComplexity("Complexity", PAC_COMPLEXITY, PAC_COMPLEXITY_SIZE, 0, "N");
#define PAC_N pacComplexity[PAC_COMPLEXITY_N]

#define PAC_SOLENOID 8
#define PAC_SOLENOID_KS 0
#define PAC_SOLENOID_SIZE 1

extern PacElemBucketKey pacSolenoid;    // pacSolenoid("Solenoid", PAC_SOLENOID, PAC_SOLENOID_SIZE, 0, "KS");
#define PAC_KS pacSolenoid[PAC_SOLENOID_KS]

#define PAC_RFCAVITY 9
#define PAC_RFCAVITY_VOLT   0
#define PAC_RFCAVITY_LAG    1
#define PAC_RFCAVITY_HARMON 2
#define PAC_RFCAVITY_SIZE   3

extern PacElemBucketKey pacRfCavity;    // pacRfCavity("RfCavity", PAC_RFCAVITY, PAC_RFCAVITY_SIZE, 1, "VOLT LAG HARMON");
#define PAC_VOLT   pacRfCavity[PAC_RFCAVITY_VOLT]
#define PAC_LAG    pacRfCavity[PAC_RFCAVITY_LAG]
#define PAC_HARMON pacRfCavity[PAC_RFCAVITY_HARMON]

#endif

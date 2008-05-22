arrange(2, 3, 0.12, 0.4, 0.25, on, off, on)

define FFTYMIN; FFTYMIN = 1
define FFTYMAX; FFTYMAX = 1000
define TAGHSH; TAGHSH = 0.01
define TAGVSH1; TAGVSH1 = 1.1
define TAGVSH2; TAGVSH2 = 2.2
define TAGVSH3; TAGVSH3 = 5.0
define NT

define t1
define t2

# Uncomment following line to hard code command file
READ BLOCK "out/test/fort.8"

# Comment header lines from fort.8 
# For now the number of lines must be a power of 2,
# i.e. 2,4,8.16,32,64,128,256,512
# For teapot fort.8 with, say, 512 turns, this requires
# commenting out either the 0 or the 512 line

# Since fft(s0,0) seems only to work with graph g0, 
# start with longit. coord. z and process it via g0

with g0
  BLOCK xy "1:6"
  NT = MAX(s0.x)
  s0.y = 1.e3*s0.y
  fft(s0,0)

move g0.s0 to g4.s0
with g4
  YAXES SCALE NORMAL
  YAXIS LABEL PLACE spec
  XAXIS LABEL "turn number"
  YAXIS LABEL "z (mm)"
  autoscale

move g0.s1 to g5.s0
with g5
  copy s0 to s1
  copy s0 to s2
  RESTRICT (s1, s0.y >= MAX(s0.y))
  define qzpk
  qzpk = s1.x[0]/NT
  # Interpolate to improve peak location, x step = Delta = 1
  #             _______                _______ 2
  # Delta_y = m*Delta_x + a*(Delta_x - Delta_x)
  #              _______
  #       a = -m/Delta_x
  #           y_+ - y_-
  #       m = ---------
  #            4*Delta
  #           Delta     y_+ - y_-
  # Delta_x = ----- * -----------------
  #             2     y_+ - 2*y_0 + y_-
  #
  define q001
  RESTRICT (s2, s2.x >= s1.x[0]-1)
  RESTRICT (s2, s2.x <= s1.x[0]+1)
  with s2
    q001 = qzpk - (y[2]-y[0])/(y[2]-2*y[1]+y[0])/2/NT
  s1 off
  s2 off

  YAXES SCALE LOGARITHMIC
  YAXIS LABEL PLACE spec
  XAXIS LABEL "tune Q\sz\N"
  YAXIS LABEL "\oy\O(Q\sz\N)"
  YAXIS LABEL PLACE spec
  WORLD YMIN FFTYMIN
  WORLD YMAX FFTYMAX
  WORLD XMIN 0.0
  WORLD XMAX QZMAX
  with s0
    x = x/NT
    y = FFTYMAX*y/MAX(y)

with g0
  BLOCK xy "1:4"
  s0.y = 1.e6*s0.y
  fft(s0,0)

move g0.s0 to g2.s0
with g2
  YAXES SCALE NORMAL
  YAXIS LABEL PLACE spec
  XAXIS LABEL "turn number"
  YAXIS LABEL "y (\f{Symbol}m\f{}m)"
  autoscale

move g0.s1 to g3.s0
with g3
  copy s0 to s1
  copy s0 to s2
  RESTRICT (s1, s0.y >= MAX(s0.y))
  define qypk
  qypk = s1.x[0]/NT
  # Interpolate to improve peak location
  define q010
  RESTRICT (s2, s2.x >= s1.x[0]-1)
  RESTRICT (s2, s2.x <= s1.x[0]+1)
  with s2
    q010 = qypk - (y[2]-y[0])/(y[2]-2*y[1]+y[0])/2/NT
  s1 off
  s2 off

  YAXES SCALE LOGARITHMIC
  YAXIS LABEL PLACE spec
  XAXIS LABEL "tune Q\sy\N"
  YAXIS LABEL "\oy\O(Q\sy\N)"
  YAXIS LABEL PLACE spec
  WORLD YMIN FFTYMIN
  WORLD YMAX FFTYMAX
  WORLD XMIN 0.0
  WORLD XMAX 0.5
  with s0
    x = x/NT
    y = FFTYMAX*y/MAX(y)

with g0
  BLOCK xy "1:2"
  s0.y = 1.e6*s0.y
  YAXES SCALE NORMAL
  YAXIS LABEL PLACE spec
  XAXIS LABEL "turn number"
  YAXIS LABEL "x (\f{Symbol}m\f{}m)"
  autoscale
  fft(s0,0)

move g0.s1 to g1.s0
with g1
  copy s0 to s1
  copy s0 to s2
  RESTRICT (s1, s0.y >= MAX(s0.y))
  define qxpk
  qxpk = s1.x[0]/NT
  # Interpolate to improve peak location
  define q100
  RESTRICT (s2, s2.x >= s1.x[0]-1)
  RESTRICT (s2, s2.x <= s1.x[0]+1)
  with s2
    q100 = qxpk - (y[2]-y[0])/(y[2]-2*y[1]+y[0])/2/NT
  s1 off
  s2 off

  YAXES SCALE LOGARITHMIC
  XAXIS LABEL "tune Q\sx\N"
  YAXIS LABEL "\ox\O(Q\sx\N)"
  YAXIS LABEL PLACE spec
  WORLD YMIN FFTYMIN
  WORLD YMAX FFTYMAX
  WORLD XMIN 0.0
  WORLD XMAX 0.5
  with s0
    x = x/NT
    y = FFTYMAX*y/MAX(y)

#-----------------------

with line
  line on
  line loctype world
  line g1
  line q100,FFTYMIN,q100,FFTYMAX
  line linewidth 0.5
  line linestyle 1
  line color 1
line def
with string
    string on
    string g1
    string loctype world
    string q100+TAGHSH, FFTYMAX*TAGVSH1
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "1 0 0"

with line
  line on
  line loctype world
  line g3
  line q100,FFTYMIN,q100,FFTYMAX
  line linewidth 0.5
  line linestyle 3
  line color 1
line def
with string
    string on
    string g3
    string loctype world
    string q100+TAGHSH, FFTYMAX*TAGVSH1
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "1 0 0"

with line
  line on
  line loctype world
  line g5
  line q100,FFTYMIN,q100,FFTYMAX
  line linewidth 0.5
  line linestyle 3
  line color 1
line def
with string
    string on
    string g5
    string loctype world
    string q100+TAGHSH, FFTYMAX*TAGVSH1
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "1 0 0"

#-----------------------

with line
  line on
  line loctype world
  line g3
  line q010,FFTYMIN,q010,FFTYMAX
  line linewidth 0.5
  line linestyle 1
  line color 1
line def
with string
    string on
    string g3
    string loctype world
    string q010+TAGHSH, FFTYMAX*TAGVSH1
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "0 1 0"

with line
  line on
  line loctype world
  line g1
  line q010,FFTYMIN,q010,FFTYMAX
  line linewidth 0.5
  line linestyle 3
  line color 1
line def
with string
    string on
    string g1
    string loctype world
    string q010+TAGHSH, FFTYMAX*TAGVSH1
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "0 1 0"

with line
  line on
  line loctype world
  line g5
  line q010,FFTYMIN,q010,FFTYMAX
  line linewidth 0.5
  line linestyle 3
  line color 1
line def
with string
    string on
    string g5
    string loctype world
    string q010+TAGHSH, FFTYMAX*TAGVSH1
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "0 1 0"

#-----------------------

with line
  line on
  line loctype world
  line g5
  line q001,FFTYMIN,q001,FFTYMAX
  line linewidth 0.5
  line linestyle 1
  line color 1
line def
with string
    string on
    string g5
    string loctype world
    string q001+TAGHSH, FFTYMAX*TAGVSH1
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "0 0 1"

with line
  line on
  line loctype world
  line g1
  line q001,FFTYMIN,q001,FFTYMAX
  line linewidth 0.5
  line linestyle 3
  line color 1
line def
with string
    string on
    string g1
    string loctype world
    string q001+TAGHSH, FFTYMAX*TAGVSH1
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "0 0 1"

with line
  line on
  line loctype world
  line g3
  line q001,FFTYMIN,q001,FFTYMAX
  line linewidth 0.5
  line linestyle 3
  line color 1
line def
with string
    string on
    string g3
    string loctype world
    string q001+TAGHSH, FFTYMAX*TAGVSH1
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "0 0 1"

#-----------------------

define q200; q200=2*q100
t1 = q200 - floor(q200)
t2 = ceil(q200) - q200
q200 = minof(t1,t2)

with line
  line on
  line loctype world
  line g1
  line q200,FFTYMIN,q200,FFTYMAX
  line linewidth 0.5
  line linestyle 6
  line color 1
line def
with string
    string on
    string g1
    string loctype world
    string q200+TAGHSH, FFTYMAX*TAGVSH2
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "2 0 0"

#-----------------------

define q300; q300=3*q100
t1 = q300 - floor(q300)
t2 = ceil(q300) - q300
q300 = minof(t1,t2)

with line
  line on
  line loctype world
  line g1
  line q300,FFTYMIN,q300,FFTYMAX
  line linewidth 0.5
  line linestyle 7
  line color 1
line def
with string
    string on
    string g1
    string loctype world
    string q300+TAGHSH, FFTYMAX*TAGVSH3
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "3 0 0"

#-----------------------

define q110; q110=q100+q010
t1 = q110 - floor(q110)
t2 = ceil(q110) - q110
q110 = minof(t1,t2)

with line
  line on
  line loctype world
  line g1
  line q110,FFTYMIN,q110,FFTYMAX
  line linewidth 0.5
  line linestyle 6
  line color 1
line def
with string
    string on
    string g1
    string loctype world
    string q110+TAGHSH, FFTYMAX*TAGVSH2
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "1 1 0"

with line
  line on
  line loctype world
  line g3
  line q110,FFTYMIN,q110,FFTYMAX
  line linewidth 0.5
  line linestyle 6
  line color 1
line def
with string
    string on
    string g3
    string loctype world
    string q110+TAGHSH, FFTYMAX*TAGVSH2
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "1 1 0"

#-----------------------

define q1m10; q1m10=q100-q010
t1 = q1m10 - floor(q1m10)
t2 = ceil(q1m10) - q1m10
q1m10 = minof(t1,t2)

with line
  line on
  line loctype world
  line g1
  line q1m10,FFTYMIN,q1m10,FFTYMAX
  line linewidth 0.5
  line linestyle 6
  line color 1
line def
with string
    string on
    string g1
    string loctype world
    string q1m10+TAGHSH, FFTYMAX*TAGVSH2
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "1 -1 0"

with line
  line on
  line loctype world
  line g3
  line q1m10,FFTYMIN,q1m10,FFTYMAX
  line linewidth 0.5
  line linestyle 6
  line color 1
line def
with string
    string on
    string g3
    string loctype world
    string q1m10+TAGHSH, FFTYMAX*TAGVSH2
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "1 -1 0"

#-----------------------

define q1p10; q1m10=q100+q010
t1 = q1p10 - floor(q1p10)
t2 = ceil(q1p10) - q1p10
q1p10 = minof(t1,t2)

with line
  line on
  line loctype world
  line g1
  line q1p10,FFTYMIN,q1p10,FFTYMAX
  line linewidth 0.5
  line linestyle 6
  line color 1
line def
with string
    string on
    string g1
    string loctype world
    string q1p10+TAGHSH, FFTYMAX*TAGVSH2
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "1 +1 0"

with line
  line on
  line loctype world
  line g3
  line q1p10,FFTYMIN,q1p10,FFTYMAX
  line linewidth 0.5
  line linestyle 6
  line color 1
line def
with string
    string on
    string g3
    string loctype world
    string q1p10+TAGHSH, FFTYMAX*TAGVSH2
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "1 +1 0"

#-----------------------

define q10m1; q10m1=q100-q001
t1 = q10m1 - floor(q10m1)
t2 = ceil(q10m1) - q10m1
q10m1 = minof(t1,t2)

with line
  line on
  line loctype world
  line g1
  line q10m1,FFTYMIN,q10m1,FFTYMAX
  line linewidth 0.5
  line linestyle 6
  line color 1
line def
with string
    string on
    string g1
    string loctype world
    string q10m1+TAGHSH, FFTYMAX*TAGVSH2
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "1 0 -1"

#-----------------------

define q10p1; q10p1=q100+q001
t1 = q10p1 - floor(q10p1)
t2 = ceil(q10p1) - q10p1
q10p1 = minof(t1,t2)

with line
  line on
  line loctype world
  line g1
  line q10p1,FFTYMIN,q10p1,FFTYMAX
  line linewidth 0.5
  line linestyle 6
  line color 1
line def
with string
    string on
    string g1
    string loctype world
    string q10p1+TAGHSH, FFTYMAX*TAGVSH2
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "1 0 +1"

#-----------------------

define q01m1; q01m1=q010-q001
t1 = q01m1 - floor(q01m1)
t2 = ceil(q01m1) - q01m1
q10m1 = minof(t1,t2)

with line
  line on
  line loctype world
  line g3
  line q01m1,FFTYMIN,q01m1,FFTYMAX
  line linewidth 0.5
  line linestyle 6
  line color 1
line def
with string
    string on
    string g3
    string loctype world
    string q01m1+TAGHSH, FFTYMAX*TAGVSH2
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "0 1 -1"

#-----------------------

define q01p1; q01p1=q010+q001
t1 = q01p1 - floor(q01p1)
t2 = ceil(q01p1) - q01p1
q1p10 = minof(t1,t2)

with line
  line on
  line loctype world
  line g3
  line q01p1,FFTYMIN,q01p1,FFTYMAX
  line linewidth 0.5
  line linestyle 6
  line color 1
line def
with string
    string on
    string g3
    string loctype world
    string q01p1+TAGHSH, FFTYMAX*TAGVSH2
    string color 1
    string rot 90
    string font 4
    string just 0
    string char size 0.5
    string def "0 1 +1"

#-----------------------

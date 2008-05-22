
PAGE SIZE 700,550

arrange(2, 3, 0.15, 0.4, 0.25, on, off, on)

define xsd; define xpsd; define ysd; define ypsd; define ctsd; define desd; 
define xav; define xpav; define yav; define ypav; define ctav; define deav; 

with g0
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "1:2"
with s0
  x = 1.0e6*x
  y = 1.0e6*y
  XAXIS LABEL "x (microns)"
  YAXIS LABEL "x' (microns/m)"
  autoscale

  xsd = SD(x)
  xav = AVG(x)
  xpsd = SD(y)
  xpav = AVG(y)
with line
  line on
  line loctype world
  line g0
  line xsd+xav, -xpsd+xpav, xsd+xav, xpsd+xpav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g0
  line -xsd+xav, -xpsd+xpav, -xsd+xav, xpsd+xpav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g0
  line -xsd+xav, xpsd+xpav, xsd+xav, xpsd+xpav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g0
  line -xsd+xav, -xpsd+xpav, xsd+xav, -xpsd+xpav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def

with g1
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "3:4"
  XAXIS LABEL "y (microns)"
  YAXIS LABEL "y' (microns/m)"
with s0
  x = 1.0e6*x
  y = 1.0e6*y
  autoscale

  ysd = SD(x)
  yav = AVG(x)
  ypsd = SD(y)
  ypav = AVG(y)
with line
  line on
  line loctype world
  line g1
  line ysd+yav, -ypsd+ypav, ysd+yav, ypsd+ypav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g1
  line -ysd+yav, -ypsd+ypav, -ysd+yav, ypsd+ypav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g1
  line -ysd+yav, ypsd+ypav, ysd+yav, ypsd+ypav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g1
  line -ysd+yav, -ypsd+ypav, ysd+yav, -ypsd+ypav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def

define xminp; define xmaxp

with g2
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "6:5"

with s0
  x = 1.0e3*x
  y = 1.0e3*y
xminp = min(x)
xmaxp = max(x)

  WORLD XMIN xminp
  WORLD XMAX xmaxp
  WORLD YMIN 0
  WORLD YMAX 50

  XAXIS LABEL "de (per mil)"
  YAXIS LABEL "dN/d(de)"

s1 on
s1 length 24
s1.x = mesh(xminp,xmaxp,24)
histogram(s0, s1.x, off, off)
# previous line does not work. Make histogram by hand

s2 on
s2 length 24
move s_ to s2
# After this instruction "s2.x" contains the mesh input to "histogram"
# which confirms that "s_" contains the output of "histogram". 
# Unfortunately "s2.x" is all zeros, which shows that $s_.y" has
# not tallied up the histogram sorting output.

# kill s0
# autoscale s2
autoticks


with g3
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "5:6"
  XAXIS LABEL "ct (mm)"
  YAXIS LABEL "de (per mil)"
with s0
  x = 1.0e3*x
  y = 1.0e3*y
  autoscale

  ctsd = SD(x)
  ctav = AVG(x)
  desd = SD(y)
  deav = AVG(y)
with line
  line on
  line loctype world
  line g3
  line ctsd+ctav, -desd+deav, ctsd+ctav, desd+deav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g3
  line -ctsd+ctav, -desd+deav, -ctsd+ctav, desd+deav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g3
  line -ctsd+ctav, desd+deav, ctsd+ctav, desd+deav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g3
  line -ctsd+ctav, -desd+deav, ctsd+ctav, -desd+deav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def


with g4
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "5:1"
  XAXIS LABEL "ct (mm)"
  YAXIS LABEL "x (microns)"
with s0
  x = 1.0e3*x
  y = 1.0e6*y
  autoscale

  xsd = SD(y)
  xav = AVG(y)
  ctsd = SD(x)
  ctav = AVG(x)
with line
  line on
  line loctype world
  line g4
  line ctsd+ctav, -xsd+xav, ctsd+ctav, xsd+xav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g4
  line -ctsd+ctav, -xsd+xav, -ctsd+ctav, xsd+xav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g4
  line -ctsd+ctav, -xsd+xav, ctsd+ctav, -xsd+xav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g4
  line -ctsd+ctav, xsd+xav, ctsd+ctav, xsd+xav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def

with g5
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "5:3"
  XAXIS LABEL "ct (mm)"
  YAXIS LABEL "y (microns)"
with s0
  x = 1.0e3*x
  y = 1.0e6*y
  autoscale

  ysd = SD(y)
  yav = AVG(y)
  ctsd = SD(x)
  ctav = AVG(x)
with line
  line on
  line loctype world
  line g5
  line ctsd+ctav, -ysd+yav, ctsd+ctav, ysd+yav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g5
  line -ctsd+ctav, -ysd+yav, -ctsd+ctav, ysd+yav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g5
  line -ctsd+ctav, -ysd+yav, ctsd+ctav, -ysd+yav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g5
  line -ctsd+ctav, ysd+yav, ctsd+ctav, ysd+yav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def

print


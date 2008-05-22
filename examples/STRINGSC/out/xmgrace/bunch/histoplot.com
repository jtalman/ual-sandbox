
PAGE SIZE 700,550

arrange(2, 3, 0.15, 0.4, 0.25, on, off, on)

define xsd; define xpsd; define ysd; define ypsd; define ctsd; define desd; 
define xav; define xpav; define yav; define ypav; define ctav; define deav; 

define nbins
nbins = 24

with g0
  TYPE XY
  BLOCK xy "5:6"
with g0.s0
  x = 1.0e3*x
  y = 1.0e3*y
  XAXIS LABEL "E_e (MeV)"
  YAXIS LABEL "dN/dE_e"
histogram(g0.s0, mesh(min(g0.s0.y),max(g0.s0.y),nbins), off, off)
move g0.s_ to g0.s2
g0.s0 off
autoscale

with g1
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "3:4"
  XAXIS LABEL "y (mm)"
  YAXIS LABEL "y' (mm/m)"
with s0
  x = 1.0e3*x
  y = 1.0e3*y
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


with g2
  TYPE XY
  BLOCK xy "6:5"
with g2.s0
  x = 1.0e3*x
  y = 1.0e3*y
  XAXIS LABEL "ct (mm)"
  YAXIS LABEL "dN/d(ct)"
histogram(g2.s0, mesh(min(g2.s0.y),max(g2.s0.y),nbins), off, off)
# Inexplicably, the histogram output goes to g0.s1
move g0.s_ to g2.s0
autoscale

with g3
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "5:6"
  XAXIS LABEL "ct (mm)"
  YAXIS LABEL "E_e (MeV)"
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
  YAXIS LABEL "x (mm)"
with s0
  x = 1.0e3*x
  y = 1.0e3*y
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
  YAXIS LABEL "y (mm)"
with s0
  x = 1.0e3*x
  y = 1.0e3*y
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


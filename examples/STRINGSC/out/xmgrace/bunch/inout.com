

PAGE SIZE 800, 550
arrange(2, 4, 0.13, 0.4, 0.25, on, off, on)
define xsd; define xpsd; define ysd; define ypsd; define ctsd; define desd; 
define xav; define xpav; define yav; define ypav; define ctav; define deav; 

READ BLOCK "bunchout"

with g0
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "1:2"
with s0
  x = 1.0e3*x
  y = 1.0e3*y
  XAXIS LABEL "x (mm)"
  YAXIS LABEL "x' (mm/m)  OUT"

  xsd = SD(x)
  xav = AVG(x)
  xpsd = SD(y)
  xpav = AVG(y)

  WORLD XMIN xav-3*xsd
  WORLD XMAX xav+3*xsd
  WORLD YMIN xpav-3*xpsd
  WORLD YMAX xpav+3*xpsd

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
  line -xsd+xav, xpsd+xpav,xsd+xav, xpsd+xpav
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

with g2
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "1:3"
  XAXIS LABEL "x (mm)"
  YAXIS LABEL "y (mm)"
with s0
  x = 1.0e3*x
  y = 1.0e3*y

  xsd = SD(x)
  xav = AVG(x)
  ysd = SD(y)
  yav = AVG(y)

  WORLD XMIN xav-3*xsd
  WORLD XMAX xav+3*xsd
  WORLD YMIN yav-3*ysd
  WORLD YMAX yav+3*ysd

with line
  line on
  line loctype world
  line g2
  line xsd+xav, -ysd+yav, xsd+xav, ysd+yav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g2
  line -xsd+xav, -ysd+yav, -xsd+xav, ysd+yav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g2
  line -xsd+xav, ysd+yav, xsd+xav, ysd+yav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g2
  line -xsd+xav, -ysd+yav, xsd+xav, -ysd+yav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def

with g4
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "5:6"
  XAXIS LABEL "ct (mm)"
  YAXIS LABEL "E_e (MeV)"
with s0
  x = 1.0e3*x
  y = 1.0e3*y

  ctsd = SD(x)
  ctav = AVG(x)
  desd = SD(y)
  deav = AVG(y)

  WORLD XMIN ctav-3*ctsd
  WORLD XMAX ctav+3*ctsd
  WORLD YMIN deav-3*desd
  WORLD YMAX deav+3*desd

with line
  line on
  line loctype world
  line g4
  line ctsd+ctav, -desd+deav, ctsd+ctav, desd+deav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g4
  line -ctsd+ctav, -desd+deav, -ctsd+ctav, desd+deav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g4
  line -ctsd+ctav, desd+deav, ctsd+ctav, desd+deav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g4
  line -ctsd+ctav, -desd+deav, ctsd+ctav, -desd+deav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def

with g6
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "5:1"
  XAXIS LABEL "ct (mm)"
  YAXIS LABEL "x (mm)"
with s0
  x = 1.0e3*x
  y = 1.0e3*y

  xsd = SD(y)
  xav = AVG(y)
  ctsd = SD(x)
  ctav = AVG(x)

  WORLD XMIN ctav-3*ctsd
  WORLD XMAX ctav+3*ctsd
  WORLD YMIN xav-3*xsd
  WORLD YMAX xav+3*xsd

with line
  line on
  line loctype world
  line g6
  line ctsd+ctav, -xsd+xav, ctsd+ctav, xsd+xav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g6
  line -ctsd, -xsd+xav, -ctsd, xsd+xav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g6
  line -ctsd+ctav, -xsd+xav, ctsd+ctav, -xsd+xav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g6
  line -ctsd+ctav, xsd+xav, ctsd+ctav, xsd+xav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def

READ BLOCK "bunchin"

with g1
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "1:2"
with s0
  x = 1.0e3*x
  y = 1.0e3*y
  XAXIS LABEL "x (mm)"
  YAXIS LABEL "x' (mm/m)  IN"

  xsd = SD(x)
  xav = AVG(x)
  xpsd = SD(y)
  xpav = AVG(y)

  WORLD XMIN xav-3*xsd
  WORLD XMAX xav+3*xsd
  WORLD YMIN xpav-3*xpsd
  WORLD YMAX xpav+3*xpsd

with line
  line on
  line loctype world
  line g1
  line xsd+xav, -xpsd+xpav, xsd+xav, xpsd+xpav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g1
  line -xsd+xav, -xpsd+xpav, -xsd+xav, xpsd+xpav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g1
  line -xsd+xav, xpsd+xpav, xsd+xav, xpsd+xpav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g1
  line -xsd+xav, -xpsd+xpav, xsd+xav, -xpsd+xpav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def

with g3
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "1:3"
  XAXIS LABEL "x (mm)"
  YAXIS LABEL "y (mm)"
with s0
  x = 1.0e3*x
  y = 1.0e3*y

  xsd = SD(x)
  xav = AVG(x)
  ysd = SD(y)
  yav = AVG(y)

  WORLD XMIN xav-3*xsd
  WORLD XMAX xav+3*xsd
  WORLD YMIN yav-3*ysd
  WORLD YMAX yav+3*ysd

with line
  line on
  line loctype world
  line g3
  line xsd+xav, -ysd+yav, xsd+xav, ysd+yav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g3
  line -xsd+xav, -ysd+yav, -xsd+xav, ysd+yav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g3
  line -xsd+xav, ysd+yav, xsd+xav, ysd+yav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g3
  line -xsd+xav, -ysd+yav, xsd+xav, -ysd+yav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def

with g5
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "5:6"
  XAXIS LABEL "ct (mm)"
  YAXIS LABEL "E_e (MeV)"
with s0
  x = 1.0e3*x
  y = 1.0e3*y

  ctsd = SD(x)
  ctav = AVG(x)
  desd = SD(y)
  deav = AVG(y)

  WORLD XMIN ctav-3*ctsd
  WORLD XMAX ctav+3*ctsd
  WORLD YMIN deav-3*desd
  WORLD YMAX deav+3*desd

with line
  line on
  line loctype world
  line g5
  line ctsd+ctav, -desd+deav, ctsd+ctav, desd+deav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g5
  line -ctsd+ctav, -desd+deav, -ctsd+ctav, desd+deav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g5
  line -ctsd+ctav, desd+deav, ctsd+ctav, desd+deav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g5
  line -ctsd+ctav, -desd+deav, ctsd+ctav, -desd+deav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def

with g7
  TYPE XY
  XAXES SCALE NORMAL
  YAXES SCALE NORMAL
  BLOCK xy "5:1"
  XAXIS LABEL "ct (mm)"
  YAXIS LABEL "x (mm)"
with s0
  x = 1.0e3*x
  y = 1.0e3*y

  xsd = SD(y)
  xav = AVG(y)
  ctsd = SD(x)
  ctav = AVG(x)

  WORLD XMIN ctav-3*ctsd
  WORLD XMAX ctav+3*ctsd
  WORLD YMIN xav-3*xsd
  WORLD YMAX xav+3*xsd

with line
  line on
  line loctype world
  line g7
  line ctsd+ctav, -xsd+xav, ctsd+ctav, xsd
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g7
  line -ctsd+ctav, -xsd+xav, -ctsd+ctav, xsd+xav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g7
  line -ctsd+ctav, -xsd+xav, ctsd+ctav, -xsd+xav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def
with line
  line on
  line loctype world
  line g7
  line -ctsd+ctav, xsd+xav, ctsd+ctav, xsd+xav
  line linewidth 2.1
  line linestyle 1
  line color 1
line def

print

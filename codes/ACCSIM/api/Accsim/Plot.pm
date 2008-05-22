package Accsim::Plot;

use strict;
use Carp;

use lib  ("$ENV{UAL_PAC}/api", "$ENV{UAL_ACCSIM}/api", "$ENV{PGPLOT}/api");
use Pac;
use PGPLOT;

sub new
{
  my $type = shift;
  my $this = {}; 

  $this->{"beamMax"}     = [0, 0, 0, 0, 0, 0];
  $this->{"beamMin"}     = [0, 0, 0, 0, 0, 0];
  $this->{"beamData"}    = [0, 0, 0, 0, 0, 0];
  $this->{"beamCounter"} = [0, 0, 0, 0, 0, 0];

  $this->{"plotLabel"}   = ["X(m)",  "PX()", 
			    "Y(m)",  "PY()", 
			    "CT(m)", "DE/P0"];

  $this->{"plotMax"}     = [0, 0, 0, 0, 0, 0];
  $this->{"plotMin"}     = [0, 0, 0, 0, 0, 0];

  $this->{"histSteps"}   = 32;
  $this->{"histBins"}    = [0, 0, 0, 0, 0, 0];
  $this->{"histBars"}    = [0, 0, 0, 0, 0, 0];
  $this->{"histMax"}     = [0, 0, 0, 0, 0, 0];

  return bless $this, $type; 
}

sub scat
{
   my ($this, $bunch, $device, $width, $ratio, $bgred, $bggreen, $bgblue, $text) = @_;
						    
   $this->defineBeamData($bunch);
   $this->definePlotSizes();    
   $this->defineHistogramBins();
   $this->defineHistogramBars();

   $this->drawPlots($device, $width, $ratio, $bgred, $bggreen, $bgblue, $text);

   $this->cleanHistogramBars();
   $this->cleanHistogramBins();
   $this->cleanPlotSizes();
   $this->cleanBeamData();     
}

sub defineBeamData
{
    my ($this, $bunch) = @_;

    my @x; my @px; my @y; my @py; my @ct, my @de;

    my ($x, $px, $y, $py, $ct, $de) = 
	(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    my $maxi =  -1.;   
    my ($xmax, $pxmax, $ymax, $pymax, $ctmax, $demax) = 
	($maxi, $maxi, $maxi, $maxi, $maxi, $maxi);

    my $mini =  +1.;
    my ($xmin, $pxmin, $ymin, $pymin, $ctmin, $demin) = 
	($mini, $mini, $mini, $mini, $mini, $mini);

    my $p;
    my ($i, $counter) = (0, 0);

    # Calculate beam sizes in the different directions

    for($i = 0; $i < $bunch->size(); $i++) {

	# reject lost particles (ACCSIM rlost)
	if($bunch->flag($i) < 1) {

	    $counter++;
	    $p = $bunch->position($i);	    

	    $x = $p->x(); push @x, $x;
	    if($x  > $xmax)   { $xmax  = $x; }
	    if($x  < $xmin)   { $xmin  = $x; }

	    $px = $p->px(); push @px, $px;
	    if($px > $pxmax)  { $pxmax = $px; }
	    if($px < $pxmin)  { $pxmin = $px; }

	    $y = $p->y(); push @y, $y;
	    if($y  > $ymax)   { $ymax  = $y; }
	    if($y  < $ymin)   { $ymin  = $y; }

	    $py = $p->py(); push @py, $py;
	    if($py > $pymax)  { $pymax = $py; }
	    if($py < $pymin)  { $pymin = $py; }

	    $ct = $p->ct(); push @ct, $ct;
	    if($ct  > $ctmax)   { $ctmax  = $ct; }
	    if($ct  < $ctmin)   { $ctmin  = $ct; }

	    $de = $p->de(); push @de, $de;
	    if($de > $demax)  { $demax = $de; }
	    if($de < $demin)  { $demin = $de; }

	}

    }
    $this->{"beamMax"}[0]     = $xmax;
    $this->{"beamMin"}[0]     = $xmin;
    $this->{"beamData"}[0]    = [@x];
    $this->{"beamCounter"}[0] = $counter; 

    $this->{"beamMax"}[1]     = $pxmax;
    $this->{"beamMin"}[1]     = $pxmin;
    $this->{"beamData"}[1]    = [@px];
    $this->{"beamCounter"}[1] = $counter; 

    $this->{"beamMax"}[2]     = $ymax;
    $this->{"beamMin"}[2]     = $ymin;
    $this->{"beamData"}[2]    = [@y];
    $this->{"beamCounter"}[2] = $counter; 

    $this->{"beamMax"}[3]     = $pymax;
    $this->{"beamMin"}[3]     = $pymin;
    $this->{"beamData"}[3]    = [@py];
    $this->{"beamCounter"}[3] = $counter;  

    $this->{"beamMax"}[4]     = $ctmax;
    $this->{"beamMin"}[4]     = $ctmin;
    $this->{"beamData"}[4]    = [@ct];
    $this->{"beamCounter"}[4] = $counter;   

    $this->{"beamMax"}[5]     = $demax;
    $this->{"beamMin"}[5]     = $demin;
    $this->{"beamData"}[5]    = [@de];
    $this->{"beamCounter"}[5] = $counter;   
}

sub cleanBeamData
{
   my $this = shift;

   my $i; 
   for($i = 0; $i < 6; $i++) {
     $this->{"beamMax"}[$i]     = 0.0;
     $this->{"beamMin"}[$i]     = 0.0;
     $this->{"beamData"}[$i]    = [];
     $this->{"beamCounter"}[$i] = 0;
   } 


}

sub definePlotSizes
{
    my $this = shift;

    my ($i, $delta) = (0, 0);
    for($i = 0; $i < 6; $i++){
      $delta = ($this->{"beamMax"}[$i] - $this->{"beamMin"}[$i])/2.;
      $this->{"plotMax"}[$i] =  $this->{"beamMax"}[$i] + $delta;
      $this->{"plotMin"}[$i] =  $this->{"beamMin"}[$i] - $delta;
    }
}

sub cleanPlotSizes
{
   my $this = shift;

   my $i;
   for($i = 0; $i < 6; $i++) {
     $this->{"plotMax"}[$i]     = 0.0;
     $this->{"plotMin"}[$i]     = 0.0;
   }
}

sub defineHistogramBins
{
  my $this = shift;
 
  my ($i, $j, $xMin, $xMax, $xBin) = (0, 0, 0, 0, 0);

  my $steps = $this->{"histSteps"};

  for($i = 0; $i < 6; $i++){ 

        my @bins;
        $xMin   = $this->{"beamMin"}[$i];
        $xMax   = $this->{"beamMax"}[$i];
        $xBin   = ($xMax-$xMin)/$steps;

        for($j = 0; $j < $steps; $j++) {
	   push @bins, $xMin  + $j*$xBin;
   	}

        $this->{"histBins"}[$i] = [@bins] ;
  }	
}

sub cleanHistogramBins
{
   my $this = shift; 
  
   my $i;
   for($i = 0; $i < 6; $i++) {
     $this->{"histBins"}[$i] = [];
   }
}

sub defineHistogramBars
{
  my $this = shift;

  my ($i, $j) = (0, 0);
  my ($xMin, $xMax, $xBin, $iBin, $barMax) = (0, 0, 0, 0, 0);
  my ($x, $bar, $xArray, $counter) = (0, 0, 0, 0);

  my $steps = $this->{"histSteps"};

  for($i = 0; $i < 6; $i++){
      
      my @bars;
      for($j = 0; $j < $steps; $j++){
	push @bars, 0.0;
      }

      $xArray  = $this->{"beamData"}[$i];
      $counter = $this->{"beamCounter"}[$i];
      $xMin    = $this->{"beamMin"}[$i];
      $xMax    = $this->{"beamMax"}[$i];
      $xBin    = ($xMax-$xMin)/$steps;
       
      foreach $x (@{$xArray}){
         $iBin = int(($x - $xMin)/$xBin);
         $bars[$iBin] += 1;
      }

      $barMax = 0;
      foreach $bar (@bars) {
	if($bar > $barMax) { $barMax = $bar; }
      }

      $this->{"histBars"}[$i] = [@bars];
      $this->{"histMax"}[$i]  = $barMax;
  }
  
}

sub cleanHistogramBars
{
   my $this = shift; 
  
   my $i;
   for($i = 0; $i < 6; $i++) {
     $this->{"histBars"}[$i] = [];
     $this->{"histMax"} = 0;
   }
}

sub drawPlots
{
    my ($this, $device, $width, $ratio, $bgred, $bggreen, $bgblue, $text) = @_;

    #changed by shishlo to show all windows
    # pgbegin(0, $device, 2, 2);  # Open plot device 
    pgopen($device);

    #define the size and shape of the graphics window
    # pgpap(width_size_in_inches, height/width_ratio) 

    pgpap($width,$ratio);    
    #set background color pgscr(CI,CR,CG,CB)
    # CI - index of the color (0-background)
    pgscr(0,$bgred,$bggreen,$bgblue);  # CR,CG,CB - red,green,blue ( 0 - 1 )

    pgsubp(2,2);              #divide window  
    #end changing

    pgscf(2);                   # Set character font
    pgslw(4);                   # Set line width
    pgsch(1.6);                 # Set character height
   
    # X-PX plot
    $this->drawPlot(0, 1, "Horizontal Plane");

    #set title for different processors
    pgsch(2.2);
    pgsci(2);
    pgscf(1);
    pgmtxt("T", 2.0, 1.0, 0., $text);
    pgsch(1.6);
    pgscf(2);

    # Y-PY plot
    $this->drawPlot(2, 3, "Vertical Plane");

    # X-Y plot
    $this->drawPlot(0, 2, "X-Y Plane");

    # CT-DE plot
    $this->drawPlot(4, 5, "Longitudinal Plane");


    # Close plot
     pgend;     
}

sub drawPlot
{
    my ($this, $i1, $i2, $title) = @_;

    # Change colour
    pgsci(4);   
            
    # Define axes
    my $pMin1 = $this->{"plotMin"}[$i1];
    my $pMax1 = $this->{"plotMax"}[$i1];
    my $pMin2 = $this->{"plotMin"}[$i2];
    my $pMax2 = $this->{"plotMax"}[$i2]; 

    if( $pMin1 == $pMax1 ) { return; }
    if( $pMin2 == $pMax2 ) { return; }  
    pgenv($pMin1, $pMax1, $pMin2, $pMax2, 0, 2); 

    # Define labels
    my $l1 = $this->{"plotLabel"}[$i1];
    my $l2 = $this->{"plotLabel"}[$i2];
    pglabel($l1, $l2, $title); 

    # Draw plot 
    my $p1 = $this->{"beamData"}[$i1];
    my $p2 = $this->{"beamData"}[$i2];
    my $counter = 
      ($this->{"beamCounter"}[$i1] < $this->{"beamCounter"}[$i2]) ?
       $this->{"beamCounter"}[$i1] : $this->{"beamCounter"}[$i2];

    pgpoint($counter, $p1, $p2, -1);

    # Change colour 
    pgsci(2);

    # Draw histograms
    $this->drawHorizontalHistogram($i1, $i2);
    $this->drawVerticalHistogram($i2, $i1);
}

sub drawHorizontalHistogram
{    
    my ($this, $i1, $i2) = @_;

    my $steps   = $this->{"histSteps"};
    my $barMax  = $this->{"histMax"}[$i1];

    my $plotMin = $this->{"plotMin"}[$i2];
    my $plotMax = $this->{"plotMax"}[$i2]; 

    my $factor = 0;
    if($barMax > 0) {
       $factor = ($plotMax - $plotMin)/$barMax/4.;
    }

    my ($bin, $binOld, $flag) = (0, 0, 0); 
    my @bins;

    foreach $bin (@{$this->{"histBins"}[$i1]}) {
        if($flag) { 
            push @bins, $binOld; 
            push @bins, $bin;
        }
        else { 
	  $flag = 1;
        }
        push @bins, $bin;
        $binOld = $bin;
    }

    my $bar;
    my @bars;

    $flag = 0;
    foreach $bar (@{$this->{"histBars"}[$i1]}) {
        if($flag) {
         push @bars, $plotMin + $bar*$factor;
         push @bars, $plotMin + $bar*$factor;
        } 
        else {
	  $flag = 1;
        }
        push @bars, $plotMin;
    }

    pgline(3*$steps - 2, \@bins, \@bars);

    # pgbin($steps, $bins, \@bars, 1);   

}

sub drawVerticalHistogram
{ 
    my ($this, $i1, $i2) = @_;

    my $steps   = $this->{"histSteps"};
    my $barMax  = $this->{"histMax"}[$i1];

    my $plotMin = $this->{"plotMin"}[$i2];
    my $plotMax = $this->{"plotMax"}[$i2]; 

    my $factor = 0;
    if($barMax > 0) {
       $factor = ($plotMax - $plotMin)/$barMax/4.;
    }

    my ($y, $yOld, $flag) = (0, 0, 0); 
    my @ys;

    foreach $y (@{$this->{"histBins"}[$i1]}) {
        if($flag) { 
            push @ys, $yOld; 
            push @ys, $y;
        }
        else { 
	  $flag = 1;
        }
        push @ys, $y;
        $yOld = $y;
    }

    my $x;
    my @xs;

    $flag = 0;
    foreach $x (@{$this->{"histBars"}[$i1]}) {
        if($flag) {
           push @xs, $plotMax - $x*$factor;
           push @xs, $plotMax - $x*$factor;
        } 
        else {
	   $flag = 1;
        }
        push @xs, $plotMax;
    }

    pgline(3*$steps - 2, \@xs, \@ys);
}

1;

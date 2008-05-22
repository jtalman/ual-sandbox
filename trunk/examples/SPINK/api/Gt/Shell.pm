package Gt::Shell;

use strict;
use Carp;
use vars qw(@ISA);

use lib ("$ENV{UAL_EXTRA}/ALE/api");
use ALE::UI::Shell;
@ISA = qw(ALE::UI::Shell);

sub new
{
  my $type = shift;
  my %params = @_;
  my $this = new ALE::UI::Shell(%params);
  return bless $this, $type;
}

# Set the beta* in the various interaction points

sub setNoc
{
  my $this   = shift;
  my $n      = shift;

  my ($g4m, $g56m, $gda, $gfa, $gfb, 
      $g7m, $g6o, $g5o, $g4o, $g6i, 
      $g5i, $g4i, $g3m, $g2m, $g1m) = @_;

  my $smfMap = $this->{"shell"}->map();

  my ($lKey, $multKey, $klKey, $kl1Key);
  $smfMap->attribKeyFromString("l",    \$lKey);
  $smfMap->bucketKeyFromString("mult",  \$multKey);
  $smfMap->attribKeyFromString("kl",    \$klKey);
  $kl1Key = $multKey->attribKey($klKey->index, 1);

  $this->_setQuad("q4o" . $n,  +$g4m, $lKey, $kl1Key);
  $this->_setQuad("q4i" . $n,  -$g4m, $lKey, $kl1Key); 
 
  $this->_setQuad("q5o" . $n,  -$g56m, $lKey, $kl1Key);
  $this->_setQuad("q5i" . $n,  +$g56m, $lKey, $kl1Key); 

  $this->_setQuad("q6o" . $n,  +$g56m, $lKey, $kl1Key);
  $this->_setQuad("q6i" . $n,  -$g56m, $lKey, $kl1Key); 

  $this->_setQuad("qda" . $n,  -$gda,  $lKey, $kl1Key); 

  $this->_setQuad("qfa" . $n,  +$gfa,  $lKey, $kl1Key); 

  $this->_setQuad("qfb" . $n,  +$gfb,  $lKey, $kl1Key); 

  $this->_setQuad("q7o" . $n,  -$g7m,  $lKey, $kl1Key);
  $this->_setQuad("q7i" . $n,  +$g7m,  $lKey, $kl1Key);

  $this->_setQuad("q6ot" . $n, +$g6o,  $lKey, $kl1Key); 

  $this->_setQuad("q5ot" . $n, -$g5o,  $lKey, $kl1Key); 

  $this->_setQuad("q4ot" . $n, +$g4o,  $lKey, $kl1Key); 

  $this->_setQuad("q6it" . $n, -$g6i,  $lKey, $kl1Key); 

  $this->_setQuad("q5it" . $n, +$g5i,  $lKey, $kl1Key); 

  $this->_setQuad("q4it" . $n, -$g4i,  $lKey, $kl1Key); 

  $this->_setQuad("q3o" . $n,  -$g3m,  $lKey, $kl1Key);
  $this->_setQuad("q3i" . $n,  +$g3m,  $lKey, $kl1Key);

  $this->_setQuad("q2o" . $n,  +$g2m,  $lKey, $kl1Key);
  $this->_setQuad("q2i" . $n,  -$g2m,  $lKey, $kl1Key);

  $this->_setQuad("q1o" . $n,  -$g1m,  $lKey, $kl1Key);
  $this->_setQuad("q1i" . $n,  +$g1m,  $lKey, $kl1Key);
   
}

sub setQuad
{
  my $this = shift;
  my ($name, $k1) = @_;

  my $smfMap = $this->{"shell"}->map();

  my ($lKey, $multKey, $klKey, $kl1Key);
  $smfMap->attribKeyFromString("l",    \$lKey);
  $smfMap->bucketKeyFromString("mult",  \$multKey);
  $smfMap->attribKeyFromString("kl",    \$klKey);
  $kl1Key = $multKey->attribKey($klKey->index, 1);

  $this->_setQuad($name,  $k1,  $lKey, $kl1Key);
}

sub setcuti
{
  my $this   = shift;
  my ($kgt, $kgti, $kqti, $kqto) = @_;

  my $smfMap = $this->{"shell"}->map();
  my ($lKey, $multKey, $klKey, $kl1Key);
  $smfMap->attribKeyFromString("l",    \$lKey);
  $smfMap->bucketKeyFromString("mult",  \$multKey);
  $smfMap->attribKeyFromString("kl",    \$klKey);
  $kl1Key = $multKey->attribKey($klKey->index, 1);

  my $kgti1 = - $kqti;
  my $kgti2 = - $kqti;
  my $kgti3 =   $kgti;
  my $kgti4 =   $kgti;
  my $kgti5 =   $kgti;
  my $kgti6 =   $kgti;
  my $kgti7 = - $kqti;
  my $kgti8 = - $kqti;

  my $kgto1 = - $kqto;
  my $kgto2 = - $kqto;
  my $kgto3 =   $kgt;
  my $kgto4 =   $kgt;
  my $kgto5 =   $kgt;
  my $kgto6 =   $kgt;
  my $kgto7 = - $kqto;
  my $kgto8 = - $kqto;

  $this->_setQuad("bo6qgt6",  $kgto1, $lKey, $kl1Key);
  $this->_setQuad("bo6qgt8",  $kgto2, $lKey, $kl1Key);
  $this->_setQuad("bo6qgt12", $kgto3, $lKey, $kl1Key);
  $this->_setQuad("bo6qgt14", $kgto4, $lKey, $kl1Key); 
  $this->_setQuad("bo6qgt16", $kgto5, $lKey, $kl1Key); 
  $this->_setQuad("bo6qgt18", $kgto6, $lKey, $kl1Key); 
  $this->_setQuad("bo7qgt8",  $kgto7, $lKey, $kl1Key);
  $this->_setQuad("bo7qgt6",  $kgto8, $lKey, $kl1Key); 

  $this->_setQuad("bi8qgt5",  $kgti1, $lKey, $kl1Key);
  $this->_setQuad("bi8qgt7",  $kgti2, $lKey, $kl1Key);
  $this->_setQuad("bi8qgt11", $kgti3, $lKey, $kl1Key);
  $this->_setQuad("bi8qgt13", $kgti4, $lKey, $kl1Key); 
  $this->_setQuad("bi8qgt15", $kgti5, $lKey, $kl1Key); 
  $this->_setQuad("bi8qgt17", $kgti6, $lKey, $kl1Key); 
  $this->_setQuad("bi9qgt7",  $kgti7, $lKey, $kl1Key);
  $this->_setQuad("bi9qgt5",  $kgti8, $lKey, $kl1Key); 

  $this->_setQuad("bo10qgt6",  $kgto1, $lKey, $kl1Key);
  $this->_setQuad("bo10qgt8",  $kgto2, $lKey, $kl1Key);
  $this->_setQuad("bo10qgt12", $kgto3, $lKey, $kl1Key);
  $this->_setQuad("bo10qgt14", $kgto4, $lKey, $kl1Key); 
  $this->_setQuad("bo10qgt16", $kgto5, $lKey, $kl1Key); 
  $this->_setQuad("bo10qgt18", $kgto6, $lKey, $kl1Key); 
  $this->_setQuad("bo11qgt8",  $kgto7, $lKey, $kl1Key);
  $this->_setQuad("bo11qgt6",  $kgto8, $lKey, $kl1Key); 

  $this->_setQuad("bi12qgt5",  $kgti1, $lKey, $kl1Key);
  $this->_setQuad("bi12qgt7",  $kgti2, $lKey, $kl1Key);
  $this->_setQuad("bi12qgt11", $kgti3, $lKey, $kl1Key);
  $this->_setQuad("bi12qgt13", $kgti4, $lKey, $kl1Key); 
  $this->_setQuad("bi12qgt15", $kgti5, $lKey, $kl1Key); 
  $this->_setQuad("bi12qgt17", $kgti6, $lKey, $kl1Key); 
  $this->_setQuad("bi1qgt7",   $kgti7, $lKey, $kl1Key);
  $this->_setQuad("bi1qgt5",   $kgti8, $lKey, $kl1Key); 

  $this->_setQuad("bo2qgt6",  $kgto1, $lKey, $kl1Key);
  $this->_setQuad("bo2qgt8",  $kgto2, $lKey, $kl1Key);
  $this->_setQuad("bo2qgt12", $kgto3, $lKey, $kl1Key);
  $this->_setQuad("bo2qgt14", $kgto4, $lKey, $kl1Key); 
  $this->_setQuad("bo2qgt16", $kgto5, $lKey, $kl1Key); 
  $this->_setQuad("bo2qgt18", $kgto6, $lKey, $kl1Key); 
  $this->_setQuad("bo3qgt8",  $kgto7, $lKey, $kl1Key);
  $this->_setQuad("bo3qgt6",  $kgto8, $lKey, $kl1Key); 

  $this->_setQuad("bi4qgt5",  $kgti1, $lKey, $kl1Key);
  $this->_setQuad("bi4qgt7",  $kgti2, $lKey, $kl1Key);
  $this->_setQuad("bi4qgt11", $kgti3, $lKey, $kl1Key);
  $this->_setQuad("bi4qgt13", $kgti4, $lKey, $kl1Key); 
  $this->_setQuad("bi4qgt15", $kgti5, $lKey, $kl1Key); 
  $this->_setQuad("bi4qgt17", $kgti6, $lKey, $kl1Key); 
  $this->_setQuad("bi5qgt7",  $kgti7, $lKey, $kl1Key);
  $this->_setQuad("bi5qgt5",  $kgti8, $lKey, $kl1Key); 


}

sub setchrom
{
  my $this   = shift;
  my ($sf0, $sd0) = @_;

  my $smfMap = $this->{"shell"}->map();
  my ($lKey, $multKey, $klKey, $kl2Key);
  $smfMap->attribKeyFromString("l",    \$lKey);
  $smfMap->bucketKeyFromString("mult",  \$multKey);
  $smfMap->attribKeyFromString("kl",    \$klKey);
  $kl2Key = $multKey->attribKey($klKey->index, 2);

  # $this->_setSext("^(bo(2|6|10)sxd1[1|3|5|7|9])\$", $sd0, $lKey, $kl2Key);
  # $this->_setSext("^(bo(3|7|11)sxd1[1|3|5|7|9])\$", $sd0, $lKey, $kl2Key);
  # $this->_setSext("^(bo(3|7|11)sxd(9|21))\$",       $sd0, $lKey, $kl2Key);

  # $this->_setSext("^(bi(4|8|12)sxd1[0|2|4|6|8])\$", $sd0, $lKey, $kl2Key);
  # $this->_setSext("^(bi(4|8|12)sxd20)\$",           $sd0, $lKey, $kl2Key);
  # $this->_setSext("^(bi[1|5|9]sxd1[0|2|4|6|8])\$",  $sd0, $lKey, $kl2Key);
  # $this->_setSext("^(bi[1|5|9]sxd20)\$",            $sd0, $lKey, $kl2Key);

  # $this->_setSext("^(bo(2|6|10)sxf1[0|2|4|6|8])\$", $sf0, $lKey, $kl2Key);
  # $this->_setSext("^(bo(2|6|10)sxf20)\$",           $sf0, $lKey, $kl2Key);
  # $this->_setSext("^(bo(3|7|11)sxf1[0|2|4|6|8])\$", $sf0, $lKey, $kl2Key);
  # $this->_setSext("^(bo(3|7|11)sxf20)\$",           $sf0, $lKey, $kl2Key);

  # $this->_setSext("^(bi(4|8|12)sxf1[1|3|5|7|9])\$", $sf0, $lKey, $kl2Key);
  # $this->_setSext("^(bi(4|8|12)sxf9)\$",            $sf0, $lKey, $kl2Key);
  # $this->_setSext("^(bi[1|5|9]sxf1[1|3|5|7|9])\$",  $sf0, $lKey, $kl2Key);
  # $this->_setSext("^(bi[1|5|9]sxf21)\$",            $sf0, $lKey, $kl2Key);

  $this->_setSext("^(sf)\$",  $sf0, $lKey, $kl2Key);
  $this->_setSext("^(sd)\$",  $sd0, $lKey, $kl2Key);
}

sub _setSext
{
  my $this      = shift;
  my ($pattern, $k2, $lKey, $kl2Key) = @_;
  
  my $it;
  for($it = $this->smf->elements->begin(); $it != $this->smf->elements->end(); $it++){
    
    if($it->second->name =~ /$pattern/) { 
 
      my $q = $it->second();

      my $l   = $q->get($lKey);
      my $kl2 = $q->get($kl2Key);
      # print $q->name, " ", $l, " ", $k2, " ", $kl2, "\n";
      my $kl2 = $l*$k2 - $kl2;

      $q->add(($kl2)*$kl2Key);
      # print $q->name, " ", $l, " ", $k2, "\n";
    }
  }

}

sub _setQuad
{
  my $this = shift;
  my ($name, $k1, $lKey, $kl1Key) = @_;
  my $it = $this->smf->elements->find($name);

  if($it != $this->smf->elements->end()) {
 
    my $q = $it->second();

    my $l   = $q->get($lKey);
    my $kl1 = $q->get($kl1Key);
    # print $name, " ", $l, " ", $k1, " ", $kl1, "\n";
    my $kl1 = $l*$k1 - $kl1;

    $q->add(($kl1)*$kl1Key);
    # print $name, " ", $l, " ", $k1, "\n";
  }
}

sub getSumL
{
  my ($this) = @_;
  my $lattice = $this->{lattice};
  my $code    = $this->{code};
	
  # Suml
  my $survey = new Pac::SurveyData;
  $code->survey($survey, 0,  $lattice->size);
  my $suml   = $survey->suml;

  return $suml;
}

1;

package PAC::MAD::Map;

use strict;
use Carp;

use lib ("$ENV{UAL_PAC}/api/");
use Pac::Smf;

# Constructors

sub new
{
    my ($type, $smf) = @_;

    my $this = {};

    $this->{elem_keys}   = {};
    $this->{bucket_keys} = {};
    $this->{attrib_keys} = {}; 

    bless $this, $type;
    $this->_initialize($smf);
 
    return $this;
}

# Element Keys

sub elemKeyFromString
{
    my ($this, $str, $elemKey) = @_;
    $$elemKey = $this->{elem_keys}->{$str};
}

sub stringFromElemKey
{
    my ($this, $elemKey, $str) = @_;
    $$str = substr($elemKey->name, 0, 4);
    $$str =~ tr/A-Z/a-z/;
}

# Bucket Keys

sub bucketKeyFromString
{
    my ($this, $str, $bucketKey) = @_;
    $$bucketKey = $this->{bucket_keys}->{$str};
}

sub stringFromBucketKey
{
    my ($this, $bucketKey, $str) = @_;
    $$str = substr($bucketKey->name, 0, 4);
    $$str =~ tr/A-Z/a-z/;
}

# Attribute Keys

sub attribKeyFromString
{
    my ($this, $str, $attribKey) = @_;
    $$attribKey = $this->{attrib_keys}->{$str};
}

sub stringFromAttribKey
{
    my ($this, $attribKey, $str) = @_;
    $$str = $attribKey->name;
    $$str =~ tr/A-Z/a-z/;
}


# ********************************************************************************************

sub _initialize
{
    my ($this, $smf) = @_;
    $this->_make_elem_keys($smf);
    $this->_make_bucket_keys($smf);
    $this->_make_attrib_keys($smf);
}

sub _make_elem_keys
{
    my ($this, $smf) = @_;

    my ($it, $key, $str);
    for($it = $smf->elemKeys->begin(); $it != $smf->elemKeys->end(); $it++){ 
	$key = $it->second;
	$this->stringFromElemKey($key, \$str);
	$this->{elem_keys}->{$str} = $key;
    } 

}

sub _make_bucket_keys
{
    my ($this, $smf) = @_;

    my ($it, $key, $str);
    for($it = $smf->bucketKeys->begin(); $it != $smf->bucketKeys->end(); $it++){
	$key = $it->second;
	$this->stringFromBucketKey($key, \$str);
	$this->{bucket_keys}->{$str} = $key;
    }
}

sub _make_attrib_keys
{
    my ($this, $smf) = @_;

    my ($it, $a, $key, $str);
    for($it = $smf->bucketKeys->begin(); $it != $smf->bucketKeys->end(); $it++){
	for($a = 0; $a < $it->second->size; $a++) {
	    $key = $it->second->attribKey($a);
	    $this->stringFromAttribKey($key, \$str);
	    $this->{attrib_keys}->{$str} = $key;
	}
    }

}

1;

__END__

=head1

=begin html
<h1> Class <a href="./package.html"> PAC::MAD</a>::Map</h1>
<hr>
<h3> Extends: </h3>
The Map class provides several methods for converting strings to and 
from SMF meta-objects.
<hr>
<pre><h3>Sample Script:  <a href="./Map.txt"> Map.pl </a> </h3></pre>
<h3> Public Methods </h3>
<ul>
<li> <b> new($smf) </b>
<dl> 
    <dt> Constructor. 
    <dd><i>smf</i> - a pointer to a SMF instance.
</dl>
<li> <b> elemKeyFromString($str, \$elemKey) </b>
<dl> 
    <dt>Converts a string, a SMF element keyword, to a PAC::SMF::ElemKey object.
    <dd><i>str</i> - a lower case four-letter string.
    <dd><i>elemKey</i> - a pointer to a PAC::SMF::ElemKey object. 
</dl>
<li> <b> stringFromElemKey($elemKey, \$str) </b>
<dl> 
    <dt> Converts a  PAC::SMF::ElemKey to a string, a SMF element keyword.
    <dd><i>elemKey</i> - a pointer to a PAC::SMF::ElemKey object. 
    <dd><i>str</i> - a lower case four-letter string.
</dl>
<li> <b> bucketKeyFromString($str, \$bucketKey) </b>
<dl> 
    <dt> Converts a string, a SMF bucket keyword, to a  PAC::SMF::ElemBucketKey object.
    <dd><i>str</i> - a lower case four-letter string.
    <dd><i>bucketKey</i> - a pointer to a PAC::SMF::ElemBucketKey object. 
</dl>
<li> <b> stringFromBucketKey($bucketKey, \$str) </b>
<dl> 
    <dt> Converts a  PAC::SMF::ElemBucketKey object to a string, a SMF bucket keyword.
    <dd><i>bucketKey</i> - a pointer to a PAC::SMF::ElemBucketKey object. 
    <dd><i>str</i> - a lower case four-letter string.
</dl>
<li> <b> attribKeyFromString($str, \$attribKey) </b>
<dl>
    <dt> Converts a string, a SMF element attribute to a  PAC::SMF::ElemAttribKey object.
    <dd><i>str</i> - a lower case  string.
    <dd><i>attribKey</i> - a pointer to a PAC::SMF::ElemAttribKey object. 
</dl>
<li> <b> stringFromAttribKey($attribKey, \$str) </b>
<dl>
    <dt> Converts a PAC::SMF::ElemAttribKey object to a string, a SMF element attribute. 
    <dd><i>attribKey</i> - a pointer to a PAC::SMF::ElemAttribKey object. 
    <dd><i>str</i> - a lower case string.
</dl>
</ul> 
<hr>

=end html

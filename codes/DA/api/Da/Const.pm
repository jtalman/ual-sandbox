package Da::Const;

use strict;
use Carp;
use vars qw(@ISA @EXPORT_OK $X_ $PX_ $Y_ $PY_ $CT_ $DE_ $PROTON_ $ELECTRON_ $INFINITY_);

require Exporter;
@ISA = qw(Exporter);
@EXPORT_OK = qw($X_ $PX_ $Y_ $PY_ $CT_ $DE_ $PROTON_ $ELECTRON_ $INFINITY_); 

*X_  = \0;
*PX_ = \1;
*Y_  = \2;
*PY_ = \3;
*CT_ = \4;
*DE_ = \5;

*PROTON_   = \0.9382796;
*ELECTRON_ = \0.5110340e-3;

*INFINITY_ = \1.0e+20;

1;

# CESR LATTICE

$smf = new Pac::Smf();

$pi = 3.14159265358979;

# Elements

# SBends

$smf->elements->declare($Sbend,"bend", "b202", "bh202", "bh203", "bh204", "b206", "b207", "bh208");

$bend->set(6.504359*$L, 0.0748*$ANGLE, 0.0450*$XSIZE, 0.0250*$YSIZE, 1.0*$SHAPE);
$bend->front->set(0.0187*$ANGLE);
$bend->end->set(0.0187*$ANGLE);

$b202->set(3.15479*$L, 0.102289*$ANGLE, 0.0450*$XSIZE, 0.0250*$YSIZE, 1.0*$SHAPE);
$bh202->set(3.15479*$L, 0.102289*$ANGLE, 0.0450*$XSIZE, 0.0250*$YSIZE, 1.0*$SHAPE);

$bh203->set(2.88008*$L, 0.020944*$ANGLE, 0.06*$XSIZE, 0.023*$YSIZE, 1.0*$SHAPE);
$bh203->front->set(0.010472*$ANGLE);
$bh203->end->set(0.010472*$ANGLE);

$bh204->set(1.578360*$L, 0.018699*$ANGLE, 0.06*$XSIZE, 0.023*$YSIZE, 1.0*$SHAPE);
$bh204->front->set(0.00935*$ANGLE);
$bh204->end->set(0.00935*$ANGLE);

$b206->set(3.22168*$L, 0.0374*$ANGLE, 0.06*$XSIZE, 0.023*$YSIZE, 1.0*$SHAPE);
$b206->front->set(0.0187*$ANGLE);
$b206->end->set(0.0187*$ANGLE);

$b207->set(3.09443*$L, 0.091254*$ANGLE, 0.0450*$XSIZE, 0.0250*$YSIZE, 1.0*$SHAPE);

$bh208->set(6.502509*$L, 0.1122*$ANGLE, 0.0450*$XSIZE, 0.0250*$YSIZE, 1.0*$SHAPE);
$bh208->front->set(0.02805*$ANGLE);
$bh208->end->set(0.02805*$ANGLE);

# Quadrupoles

$smf->elements->declare($Quadrupole, "acq43", "acq49", "qadd", "quad", "q1", "q1e", "q2", "q2e",  
			"q48", "q49", "reqw", "reqe", "skq2", "skqv2", "skqv3", "skqv4", "skqv5", 
			"skq7", "skq14", "skq29", "skq47", "skq48");

$acq43->set( 0.49*$L );
$acq49->set( 0.50*$L );

$qadd->set( 0.6*$L, 0.0450*$XSIZE, 0.0250*$YSIZE, 1.0*$SHAPE, 1.0*$N);

$quad->set( 0.558800*$L, 0.0450*$XSIZE, 0.0250*$YSIZE, 1.0*$SHAPE,  1.0*$N );
$q1->set(   0.899160*$L, 0.0525*$XSIZE, 0.0250*$YSIZE, 1.0*$SHAPE,
	 (6.8659*$pi/180.)*$TILT, 10.0*$N );
$q1e->set(  0.899160*$L, 0.0525*$XSIZE, 0.0250*$YSIZE, 1.0*$SHAPE,
	 (-6.8659*$pi/180.)*$TILT, 10.0*$N );
$q2->set(   0.548640*$L, 0.0650*$XSIZE, 0.0455*$YSIZE, 1.0*$SHAPE, 
	 (14.5544*$pi/180.)*$TILT, 10.0*$N );
$q2e->set(   0.548640*$L, 0.0650*$XSIZE, 0.0455*$YSIZE, 1.0*$SHAPE, 
	 (-14.5544*$pi/180.)*$TILT, 10.0*$N );
$q48->set(  0.548640*$L, 0.0650*$XSIZE, 0.0455*$YSIZE, 1.0*$SHAPE, 1.0*$N );
$q49->set(  0.899916*$L, 0.0525*$XSIZE, 0.0250*$YSIZE, 1.0*$SHAPE, 1.0*$N );
  
$reqw->set( 1.524800*$L, (-0.8376*1.5248)*$KL1, (-0.085*0.9)*$KS, 0.0450*$XSIZE, 0.0250*$YSIZE, 1.0*$SHAPE, 
	  ( 4.4884*$pi/180.)*$TILT, 10.0*$N);
$reqe->set( 1.524800*$L, (-0.8376*1.5248)*$KL1, (-0.085*0.9)*$KS, 0.0450*$XSIZE, 0.0250*$YSIZE, 1.0*$SHAPE, 
	  (-4.4884*$pi/180.)*$TILT, 10.0*$N);

$skq2->set(  0.1727*$L );
$skqv2->set( 0.1727*$L );
$skqv3->set( 0.1727*$L );
$skqv4->set( 0.1727*$L );
$skqv5->set( 0.1727*$L );
$skq7->set(  0.1524*$L );
$skq14->set( 0.4075*$L );
$skq29->set( 0.408*$L );
$skq47->set( 0.128*$L );
$skq48->set( 0.254*$L );

# Sextupoles

$smf->elements->declare($Sextupole, "skx", "skxe", "x", "xv");

$skx->set( 0.1285*$L );
$skxe->set( 0.1262*$L );
$x->set( 0.0450*$XSIZE, 0.0250*$YSIZE, 1.0*$SHAPE );
$xv->set( 0.0450*$XSIZE, 0.0250*$YSIZE, 1.0*$SHAPE );

# Octupoles

$smf->elements->declare($Octupole,"oct");

$oct->set( 0.381*$L );

# Monitors

$smf->elements->declare($Monitor, "dt", "dtx");

# Hkickers

$smf->elements->declare($Hkicker, "pbh26", "pbh28", "pbh35", "pbh35e", "pbh38e", "h7", "h8", "pih", "pihe",  "sph7", "sph44", "sfh");

$pbh26->set( 0.2295*$L );
$pbh28->set( 0.2295*$L );
$pbh35->set( 0.2250*$L );
$pbh35e->set( 0.2302*$L );
$pbh38e->set( 0.2290*$L );

$h7->set( 0.25*$L );
$h8->set( 0.255*$L );

$pih->set( 0.229*$L );
$pihe->set( 0.5112*$L );

$sph7->set( 2.442*$L );
$sph44->set( 2.3894*$L );

$sfh->set( 0.254*$L ); # Horz WB Feedback Shaker

# Vkickers

$smf->elements->declare($Vkicker, "v7", "v7e", "v9", "v16", "v47", "v48", "piv", "spv", "sfv");

$v7->set( 0.127*$L );
$v7e->set( 0.254*$L );
$v9->set( 0.255*$L );
$v16->set( 0.25*$L );
$v47->set( 0.254*$L );
$v48->set( 0.254*$L );

$piv->set( 0.254*$L );
$spv->set( 2.647*$L );

$sfv->set( 0.254*$L ); # Vert NB Feedback Shaker

# Kickers

$smf->elements->declare($Kicker, hv1, kvh1, kvh2, sbhv);

$hv1->set( 0.1727*$L );

$kvh1->set( 1.2785*$L ); 
$kvh2->set( 1.2775*$L );

$sbhv->set( 0.23*$L );

# RfCavity

$smf->elements->declare($RfCavity, rf);

$rf->set( 1.8*$L );

# Drifts

$smf->elements->declare($Drift, "cer", "cere");

for($i = 0; $i < 50; $i++){
    $smf->elements->declare($Drift, "d$i");
} 

$cer->set( 0.254*$L );
$cere->set( 0.54*$L ); 

# Other elements

$smf->elements->declare($Element, "dfh", "dfv", "dtc", "dtf", "dti", "dtv",
			"elm", "ijs", "msk", "plm", "pls", "wig", "tdd", "sch", "scv", "sld");

$dtc->set( 0.352*$L ); # Intensity detector

$ijs->set( 1.015*$L ); # septum will be implemented as a WILD element (separate class)

$wig->set( 2.355*$L ); # wiggler will be implemented as a COSY element (DA Integrator)

$sld->set( 0.3*$L );   # Strip line beam detector

# Markers

for($i = 0; $i < 50; $i++){
    $smf->elements->declare($Marker, "m$i");
} 

$smf->elements->declare($Marker, "mhalf");
			
# Lines

$smf->lines->declare("half1", "half2", "whalf", "ehalf");
 
for($i = 0; $i < 50; $i++){
    $smf->lines->declare("cell$i");
} 

$smf->lines->declare("cell0e",  "cell1e",  "cell2e",  "cell7e",  "cell8e",  "cell9e",
		     "cell10e", "cell11e", "cell12e", "cell14e", "cell16e", "cell18e",
                     "cell23e", "cell30e", "cell32e", "cell33e", "cell35e", "cell38e", 
		     "cell43e", "cell46e", "cell48e");


$cell0->set( $m0, $d0, $msk, $d0, $dt, $d0, $reqw, $d0, $dt, $d0); 
$cell0e->set($m0, $d0, $msk, $d0, $dt, $d0, $reqe, $d0, $dt, $d0);
 
$cell1->set( $m1, $q1,  $d1, $hv1, $d1);
$cell1e->set($m1, $q1e, $d1, $hv1, $d1);

$cell2->set( $m2, $q2,  $d2, $dt,  $d2, $dt, $d2, $skq2, $d2, $skqv2, $d2, $bh203, $d2, $bh204, $d2, $sch, $d2, $dt, $d2);
$cell2e->set($m2, $q2e, $d2, $dt,  $d2, $dt, $d2, $skq2, $d2, $skqv2, $d2, $bh203, $d2, $bh204, $d2, $sch, $d2, $dt, $d2);

$cell3->set($m3, $quad, $d3, $bh202, $d3, $skqv3, $d3, $dt, $d3);
$cell4->set($m4, $quad, $d4, $b202, $d4, $skqv4, $d4, $dt, $d4);
$cell5->set($m5, $quad, $d5, $b202, $d5, $skqv5, $d5, $dt, $d5);
$cell6->set($m6, $quad, $d6, $b202, $d6); 
             
$cell7->set( $m7, $quad, $d7, $dt, $d7, $v7,  $d7, $dti,$d7, $wig, $d7, $h7, $d7,  $skq7, $d7, $sph7,$d7);
$cell7e->set($m7, $quad, $d7, $dt, $d7, $skxe,$d7, $v7e,$d7, $wig, $d7, $h7, $d7,  $skq7, $d7, $sph7,$d7); # dth

$cell8->set( $m8, $quad, $d8, $xv, $d8, $dt, $d8,           $rf, $d8, $rf, $d8, $dt, $d8);
$cell8e->set($m8, $quad, $d8, $xv, $d8, $dt, $d8, $h8, $d8, $rf, $d8, $rf, $d8, $dt, $d8);

$cell9->set( $m9, $quad, $d9, $xv, $d9, $v9, $d9, $dtv,$d9, $dfv,$d9, $dti,$d9, $kvh1, $d9, $kvh2, $d9, $dtx, $d9, $dfh, $d9);
$cell9e->set($m9, $quad, $d9, $xv, $d9); 

$cell10->set( $m10, $quad, $d10, $x, $d10, $dt, $d10, $dfv, $d10, $bend, $d10);
$cell10e->set($m10, $quad, $d10, $x, $d10, $dt,             $d10, $bend, $d10);

$cell11->set( $m11, $quad, $d11, $xv, $d11, $dt, $d11, $dfh, $d11, $bend, $d11);
$cell11e->set($m11, $quad, $d11, $xv, $d11, $dt,             $d11, $bend, $d11);

$cell12->set( $m12, $quad, $d12, $x, $d12, $dt, $d12, $tdd, $d12, $bend, $d12);
$cell12e->set($m12, $quad, $d12, $x, $d12, $dt,             $d12, $bend, $d12);

$cell13->set($m13, $quad, $d13, $xv, $d13, $dt, $d13, $bend, $d13);

$cell14->set( $m14, $quad, $d14, $x, $d14, $dt, $d14, $skq14, $d14, $elm, $d14, $bend, $d14);
$cell14e->set($m14, $quad, $d14, $x, $d14, $dt, $d14, $skq14,             $d14, $bend, $d14);

$cell15->set($m15, $quad, $d15, $xv, $d15, $dt, $d15, $bend, $d15); # sch

$cell16->set( $m16, $quad, $d16, $x, $d16, $dt, $d16, $v16, $d16, $bend, $d16);
$cell16e->set($m16, $quad, $d16, $x, $d16, $dt, $d16,             $bend, $d16);

$cell17->set($m17, $quad, $d17, $xv, $d17, $dt, $d17, $bend, $d17);

$cell18->set( $m18, $quad, $d18, $x, $d18, $dt, $d18, $dtx, $d18);
$cell18e->set($m18, $quad, $d18, $x, $d18, $dt, $d18, $dti, $d18);

$cell19->set($m19, $quad, $d19, $xv, $d19, $dt, $d19, $bend, $d19);

$cell20->set($m20, $quad, $d20, $x, $d20, $dt, $d20, $bend, $d20);
$cell21->set($m21, $quad, $d21, $xv, $d21, $dt, $d21, $bend, $d21);  
$cell22->set($m22, $quad, $d22, $x, $d22, $dt, $d22, $bend, $d22);

$cell23->set( $m23, $quad, $d23, $xv, $d23, $dt, $d23, $skx, $d23, $dtx, $d23, $piv, $d23,              $bend, $d23);
$cell23e->set($m23, $quad, $d23, $xv, $d23, $dt, $d23, $skxe,$d23, $plm, $d23, $dtf, $d23, $sbhv, $d23, $bend, $d23);

$cell24->set($m24, $quad, $d24, $x, $d24, $dt, $d24, $bend, $d24);
$cell25->set($m25, $quad, $d25, $xv, $d25, $dt, $d25, $bend, $d25);
$cell26->set($m26, $quad, $d26, $x, $d26, $dt, $d26, $pbh26, $d26, $bend, $d26);
$cell27->set($m27, $quad, $d27, $xv, $d27, $dt, $d27, $bend, $d27);
$cell28->set($m28, $quad, $d28, $x, $d28, $dt, $d28, $pbh28, $d28, $bend, $d28);
$cell29->set($m29, $quad, $d29, $xv, $d29, $dt, $d29, $skq29, $d29, $bend, $d29);

$cell30->set( $m30, $quad, $d30, $x, $d30, $dt,             $d30, $bend, $d30); 
$cell30e->set($m30, $quad, $d30, $x, $d30, $dt, $d30, $sfv, $d30, $bend, $d30); 

$cell31->set($m31, $quad, $d31, $xv, $d31, $dt, $d31, $bend, $d31);

$cell32->set( $m32, $quad, $d32, $x, $d32,       $dt, $d32,             $bend, $d32);
$cell32e->set($m32, $quad, $d32, $x, $d32, $dfh, $dt, $d32, $sfh, $d32, $bend, $d32);

$cell33->set( $m33, $quad, $d33, $xv, $d33, $dt, $d33, $dtx, $d33, $ijs, $d33);
$cell33e->set($m33, $quad, $d33, $xv, $d33, $dt,             $d33, $ijs, $d33);

$cell34->set($m34, $quad, $d34, $x, $d34, $bend, $d34);

$cell35->set( $m35, $quad, $d35, $xv, $d35, $dt, $d35, $bend, $d35, $dti, $d35, $dti, $d35, $cer, $d35, $pih,  $d35, $pbh35, $d35, $sch, $d35);
$cell35e->set($m35, $quad, $d35, $xv, $d35, $dt, $d35, $bend, $d35, $sld,             $d35, $cere,$d35, $pihe, $d35, $pbh35e,$d35, $sch, $d35);

$cell36->set($m36, $quad, $d36, $x, $d36, $dt, $d36, $bend, $d36);
$cell37->set($m37, $quad, $d37, $xv, $d37, $dt, $d37, $bend, $d37);

$cell38->set( $m38, $quad, $d38, $x, $d38, $dt, $d38, $bend, $d38);
$cell38e->set($m38, $quad, $d38, $x, $d38, $dt, $d38, $pbh38e, $d38, $bend, $d38);

$cell39->set($m39, $quad, $d39, $xv, $d39, $dt, $d39, $bend, $d39);
$cell40->set($m40, $quad, $d40, $x, $d40, $dt, $d40, $bend, $d40);
$cell41->set($m41, $quad, $d41, $xv, $d41, $dt, $d41, $bend, $d41);
$cell42->set($m42, $quad, $d42, $x, $d42, $dt, $d42, $bend, $d42);

$cell43->set( $m43, $quad, $d43, $xv, $d43, $dt, $d43, $scv, $d43, $acq43, $d43, $dtv,  $d43, $dtx, $d43, $bh208, $d43);
$cell43e->set($m43, $quad, $d43, $xv, $d43, $dt, $d43, $scv, $d43, $acq43, $d43, $dtv,  $d43, $dti, $d43, $bh208, $d43);

$cell44->set($m44, $quad, $d44, $x, $d44, $dt, $d44, $sph44, $d44, $oct, $d44);
$cell45->set($m45, $quad, $d45, $xv, $d45, $dt, $d45, $bh208, $d45);

$cell46->set( $m46, $quad, $d46, $dt, $d46, $dtc, $d46, $b206, $d46);
$cell46e->set($m46, $quad, $d46, $dt, $d46,             $b206, $d46);

$cell47->set( $m47, $quad, $d47, $skq47, $d47, $b207, $d47, $qadd, $d47, $dt, $d47, $bh203, $d47, $spv, $d47, $dt, $d47); # v47 ?

$cell48->set( $m48, $q48,  $d48, $v48, $d48, $sch, $d48, $oct, $d48, $skq48, $d48); 
$cell48e->set($m48, $q48,  $d48, $v48, $d48, $sch, $d48, $oct, $d48, $skq48, $d48, $scv, $d48); 

$cell49->set($m49, $q49,  $d49, $dt,  $d49, $acq49, $d49);

$half1->set($cell0,  $cell1,  $cell2,  $cell3,  $cell4,  $cell5,  $cell6,  $cell7,  $cell8,  $cell9,
	    $cell10, $cell11, $cell12, $cell13, $cell14, $cell15, $cell16, $cell17, $cell18, $cell19,
	    $cell20, $cell21, $cell22, $cell23, $cell24, $cell25, $cell26, $cell27, $cell28, $cell29,
	    $cell30, $cell31, $cell32, $cell33, $cell34, $cell35, $cell36, $cell37, $cell38, $cell39,
	    $cell40, $cell41, $cell42, $cell43, $cell44, $cell45, $cell46, $cell47, $cell48, $cell49, 
	    $mhalf);

$half2->set($cell0e, $cell1e, $cell2e, $cell3,  $cell4,  $cell5,  $cell6,  $cell7e, $cell8e, $cell9e,
	    $cell10e,$cell11e,$cell12e,$cell13, $cell14e,$cell15, $cell16e,$cell17, $cell18e,$cell19,
            $cell20, $cell21, $cell22, $cell23e,$cell24, $cell25, $cell26, $cell27, $cell28, $cell29,
	    $cell30e,$cell31, $cell32e,$cell33e,$cell34, $cell35e,$cell36, $cell37, $cell38e,$cell39,
	    $cell40, $cell41, $cell42, $cell43e,$cell44, $cell45, $cell46e,$cell47, $cell48e,$cell49);

$whalf->set($half1);
$ehalf->set((-1.)*$half2);

# Lattices

$smf->lattices->declare("west", "east", "ring");

$west->set($whalf);
$east->set($ehalf);
$ring->set($west, $east);

# Flat instantiation

# Sextupoles

@west_xlengths  = (
    0.128,  0.128,  0.1292,                 #  8 - 10
    0.128,  0.1281, 0.1268, 0.1275, 0.129,  # 11 - 15
    0.1285, 0.1288, 0.1284, 0.1278, 0.1285, # 16 - 20
    0.1275, 0.1274, 0.128,  0.1275, 0.127,  # 21 - 25
    0.128,  0.1275, 0.128,  0.1265, 0.127,  # 26 - 30
    0.127,  0.1275, 0.128,  0.128,  0.1275, # 31 - 35
    0.1275, 0.127,  0.127,  0.128,  0.128,  # 36 - 40
    0.128,  0.128,  0.127,  0.128,  0.1276  # 41 - 45
);

@indexes = $west->indexes("^(xv?)");
for($i = 0; $i < @indexes; $i++){
    $west->element($indexes[$i])->set( $west_xlengths[$i]*$L );
}

@east_xlengths  = (
    0.128,  0.128,  0.1262,                  #  8 - 10
    0.1275, 0.1275, 0.1275, 0.1274, 0.1278,  # 11 - 15
    0.1278, 0.1274, 0.1278, 0.1262, 0.127,   # 16 - 20
    0.1282, 0.1258, 0.127,  0.1274, 0.1262,  # 21 - 25
    0.1262, 0.1266, 0.1262, 0.1282, 0.1282,  # 26 - 30
    0.1274, 0.1286, 0.1274, 0.127,  0.1275,  # 31 - 35
    0.1266, 0.127,  0.1258, 0.1278, 0.127,   # 36 - 40
    0.1278, 0.1274, 0.1278, 0.1274, 0.127    # 41 - 45
		   );

@indexes = $east->indexes("^(xv?)");
$esize = @indexes;
for($i = 0; $i < $esize; $i++){
    $east->element($indexes[$i])->set( $east_xlengths[$esize - $i - 1]*$L );
}

# Drifts

@west_dlengths  = (
      0.260001, 0.3623, 0.0177, 0.32862, 0.457,                             #  0
      1.0878, 0.248,                                                        #  1
      0.12, 0.36, 1.732, 0.1073, 0.464897, 0.442409, 0.472652, 0.11, 0.08,  #  2
      0.2975365, 0.2687565, 0.258, 0.075,                                   #  3 0.297536, 0.268757
      0.5573075, 0.2687565, 0.2547, 0.0783,                                 #  4
      0.557307, 0.2687565, 0.2577, 0.0753,                                  #  5
      0.5573065, 0.360166,                                                  #  6
      0.121, 0.057, 0.539, 0.35, 0.11, 0.087, 0.35525, 0.485,               #  7
      0.154, 0.175, 1.275, 2.09153, 1.461, 0.025,                           #  8
      0.162, 0.39, 0.005, 0.0, 1.075, 0.31327, 0.1845, 0.377, 0.101, 0.405, #  9
      0.1564, 0.1223, 0.0, 0.5206915, 0.2126915,                            # 10
      0.156, 0.1207, 0.0, 0.5238925, 0.2126915,                             # 11
      0.1567, 0.0985, 0.3297, 0.6755915, 0.2126915,                         # 12
      0.1617, 0.1184, 0.5216915, 0.2126915,                                 # 13
      0.158, 0.0991, 0.5574, 0.3605, 1.1385915, 0.212691,                   # 14 
      0.157, 0.1191, 0.5234915, 0.2126915,                                  # 15 
      0.159, 0.1122, 0.0003, 0.2785925, 0.2126915,                          # 16 
      0.1555, 0.1218, 0.5224915, 0.2126915,                                 # 17
      0.15, 0.1033, 5.5797, 0.1398,                                         # 18
      0.1556, 0.1136,  0.5315915, 0.212692,                                 # 19
      0.1587, 0.1179, 0.5234915, 0.2126915,                                 # 20
      0.1567, 0.1209, 0.5234915, 0.2126915,                                 # 21
      0.159, 0.1179, 0.5242915, 0.2126915,                                  # 22
      0.158, 0.0972, 0.5548, 1.0062, 0.2272, 0.3746925, 0.2126915,          # 23
      0.1564, 0.1208, 0.5238915, 0.2126915,                                 # 24
      0.158, 0.1197, 0.5238915, 0.2126915,                                  # 25  
      0.1595, 0.0958, 0.1413, 0.3994915, 0.2126915,                         # 26
      0.16, 0.1202, 0.5208915, 0.2126915,                                   # 27
      0.1585, 0.0987, 0.1432, 0.3956925, 0.2126915,                         # 28
      0.158, 0.0993, 0.7032, 0.5135915, 0.2126915,                          # 29
      0.1575, 0.0983, 0.7707915, 0.2126915,                                 # 30
      0.1595, 0.1197, 0.5223915, 0.2126915,                                 # 31
      0.154, 0.0975, 0.7745915, 0.2126915,                                  # 32
      0.157, 0.0988, 3.0707, 0.6517, 0.29,                                  # 33
      0.1295, 0.3663515, 0.2126915,                                         # 34
      0.157, 0.1042, 0.6923525, 0.4437515, 0.755, 0.1164, 0.1789, 0.1784, 
      0.28, 0.166,                                                          # 35
      0.158, 0.0973, 0.3562915, 0.2126915,                                  # 36
      0.1585, 0.1182, 0.5248915, 0.2126915,                                 # 37
      0.1565, 0.0993, 0.7707915, 0.2126915,                                 # 38
      0.156, 0.1207, 0.5238915, 0.2126915,                                  # 39
      0.1575, 0.1197, 0.5233925, 0.2126915,                                 # 40
      0.158, 0.1182, 0.5243915, 0.2126915,                                  # 41
      0.1585, 0.0958, 0.5462915, 0.2126915,                                 # 42
      0.1545, 0.1028, 0.2165, 0.2342, 0.1682, 1.7655, 0.4438895, 0.239589,  # 43
      0.1557, 0.0991, 0.853, 0.711810, 1.063,                               # 44
      0.1573, 0.2001, 0.4450305, 0.2395895,                                 # 45      
      0.1131, 0.4767, 0.437546, 0.907145,                                   # 46
      0.155, 0.884636, 0.543146, 0.113747, 0.709, 0.369288, 0.189, 0.225,   # 47
      0.1, 0.121, 0.065, 0.538722, 0.387,                                   # 48
      0.21, 1.18, 0.135078                                                  # 49
  );

@indexes = $west->indexes("^(d[0-9]+)");
for($i = 0; $i < @indexes; $i++){
    $west->element($indexes[$i])->set( $west_dlengths[$i]*$L );
}

# west vs east

$dh7 = (0.25 - 0.245)/2.;
$dskq7 = (0.1524 - 0.152)/2.;
$dskq14 = (0.4075 - 0.4078)/2.;
$dpbh26 = (0.2295 - 0.2286)/2.;
$dpbh28 = (0.2295 - 0.2286)/2.;
$dskq29 = (0.408 - 0.4077)/2.;
$dskq47 = (0.128 - 0.125)/2.;


@east_dlengths  = (
      0.260004, 0.36232, 0.0177, 0.36162, 0.424,                            #  0
      1.0968, 0.239,                                                        #  1
      0.116, 0.349, 1.747, 0.1083, 0.463897, 0.442409, 0.467152, 0.116, 0.0795, # 2
      0.297537, 0.284756, 0.2423, 0.0747,                                   #  3
      0.557307, 0.269057, 0.2507, 0.082,                                    #  4
      0.557306, 0.284757, 0.229, 0.088,                                     #  5
      0.557306, 0.360167,                                                   #  6
      0.237, 0.048, 0.2308, 0.374, 0.03 - $dh7, 0.046 - $dskq7 - $dh7, 0.40465 - $dskq7, 0.486, #  7
      0.1572, 0.1698, 0.005, 1.02653, 2.096, 1.342, 0.13,                   #  8
      0.1565, 5.66727,                                                      #  9 
      0.1575, 0.1231, 0.521791, 0.212692,                                   # 10
      0.1576, 0.1217, 0.521792, 0.212692,                                   # 11
      0.156,  0.1007, 1.004391, 0.212692,                                   # 12
      0.1572, 0.1221, 0.521791, 0.212692,                                   # 13
      0.1563, 0.0985, 0.5795 - $dskq14, 1.479091 - $dskq14, 0.212692,       # 14
      0.1549, 0.1214, 0.524491, 0.212692,                                   # 15
      0.1565, 0.1221, 0.522192, 0.212692,                                   # 16
      0.1598, 0.1171, 0.524291, 0.212692,                                   # 17
      0.1554, 0.1003, 5.5764,   0.1413,                                     # 18
      0.1565, 0.0963, 0.549591, 0.212692,                                   # 19
      0.1603, 0.1193, 0.521991, 0.212692,                                   # 20
      0.1595, 0.1174, 0.523491, 0.212692,                                   # 21
      0.1577, 0.1232, 0.521891, 0.2126915,                                  # 22
      0.1567, 0.1014, 0.526, 0.4057, 0.6316, 0.2322, 0.391793, 0.212691,    # 23
      0.158,  0.1205, 0.522692, 0.2126915,                                  # 24
      0.1577, 0.1235, 0.521191, 0.212692,                                   # 25
      0.157,  0.1006, 0.1453 - $dpbh26, 0.395891 - $dpbh26, 0.2126915,      # 26
      0.1577, 0.1229, 0.521392, 0.212691,                                   # 27
      0.1585, 0.0978, 0.1486 - $dpbh28, 0.393893 - $dpbh28, 0.2126915,      # 28
      0.1572, 0.0986, 0.6749 - $dskq29, 0.541991 - $dskq29, 0.2126915,      # 29                      
      0.1575, 0.0978, 0.1372, 0.378892, 0.212691,                           # 30
      0.1572, 0.12, 0.5239915, 0.212692,                                    # 31
      0.1537, 0.1012, 0.1372, 0.378891, 0.212692,                           # 32
      0.1572, 0.0977, 3.7239, 0.29,                                         # 33
      0.1359, 0.360951, 0.212692,                                           # 34
      0.1544, 0.1088, 0.690352, 0.525252, 0.1652, 0.0923, 0.119,
      0.1766, 0.1667,                                                       # 35
      0.157, 0.0989, 0.356591,  0.212692,                                   # 36
      0.1572, 0.121, 0.523391,  0.212692,                                   # 37
      0.1554, 0.1158, 0.129,  0.398591, 0.212692,                           # 38
      0.158,  0.1191, 0.523691, 0.212692,                                   # 39
      0.158, 0.1199, 0.523691,  0.212693,                                   # 40
      0.1565, 0.1208, 0.523491, 0.212692,                                   # 41
      0.1539, 0.1097, 0.5375915, 0.2126915,                                 # 42
      0.1567, 0.0985, 0.2158, 0.2272, 0.1735, 1.7655, 0.44759, 0.239589,    # 43
      0.157, 0.0983, 0.83961, 0.520300, 1.268000,                           # 44
      0.157, 0.1883, 0.457729, 0.239589,                                    # 45
      0.1093, 1.270046, 0.907147,                                           # 46
      0.155 - $dskq47, 0.887636 - $dskq47, 0.543146, 0.120547, 0.7022, 0.369288, 0.194, 0.22,    # 47
      0.042, 0.204, 0.11, 0.465722, 0.14, 0.25,                             # 48
      0.2, 1.19, 0.135054                                                   # 49
		   );

@indexes = $east->indexes("^(d[0-9]+)");
$esize = @indexes;
for($i = 0; $i < $esize; $i++ ){
    $east->element($indexes[$i])->set( $east_dlengths[$esize - $i - 1]*$L );
}

# Also add misalignments, field errors, rms uncertainties

# Add  "external" elements such as IR solenoid

@indexes = $west->indexes("^(d0)");
for($i = 0; $i < 3; $i++){
    $west->element($indexes[$i])->add((-0.085*0.9)*$KS);
}

@indexes = $east->indexes("^(d0)");
$esize = @indexes;
for($i = 0; $i < 3; $i++){
    $east->element($indexes[$esize - $i - 1])->add((-0.085*0.9)*$KS);
}

1;

# CESR Control History Set (CESR HS).
# SMF will be connected with the CESR HS via Migrator.  Here we show  
# the correspondence between  SMF elements and  HS data. You can see 
# that many elements require the superposition of element attributes.

# CSR HSP VOLT

@indexes = $west->indexes("^(sph[0-9]+)");

$cra = 0.0;
$sph7->set((0.000068235/2.62*$cra)*$KL0);

# CSR HORZ CUR

@indexes = $west->indexes("^(bend|bh[0-9]+|hv1|h7)");

# CSR VERT CUR

@indexes = $west->indexes("^(xv|v[0-9]+|hv1|skqv[0-9]+)");

# CSR QUAD CUR

$mlength = 0.6;
@quad_k1 = ( # from Teapot file
     0.665606*0.95/0.6,   -0.232746, -0.296364,  0.410842, -0.288915,  #  1 -  5
     0.408156,            -0.264821,  0.141031, -0.218935,  0.229558,  #  6 - 10
    -0.210051,             0.228398, -0.243352,  0.271697, -0.232716,  # 11 - 15
     0.267593,            -0.261477,  0.267542, -0.328507,  0.245163,  # 16 - 20
    -0.285077,             0.256764, -0.234584,  0.244626, -0.294958,  # 21 - 25
     0.259367,            -0.293133,  0.249048, -0.228843,  0.257508,  # 26 - 30 
    -0.200739,             0.207124, -0.201227,  0.250069, -0.182243,  # 31 - 35
     0.235810,            -0.238413,  0.260686, -0.217470,  0.255270,  # 36 - 40
    -0.212221,             0.241472, -0.190566,  0.292308, -0.271935,  # 41 - 45
     0.336803,            -0.052574,  0.546159, (-0.272314)*0.95/0.6   # 46 - 49
);

@indexes = $west->indexes("^(quad|q[0-9]+)");
for($i = 0; $i < @indexes; $i++){
    $west->element($indexes[$i])->add( ($quad_k1[$i]*$mlength)*$KL1 );
}

@indexes = $east->indexes("^(quad|q[0-9]+|q1e|q2e)");
$esize = @indexes;
for($i = 0; $i < $esize; $i++){
    $east->element($indexes[$i])->add( ($quad_k1[$esize - $i - 1]*$mlength)*$KL1 );
}

# CSR QADD CUR

@indexes = $west->indexes("^(qadd)"); # KL1

   $west->element($indexes[0])->add( (-0.268153*0.6)*$KL1 );

@indexes = $east->indexes("^(qadd)"); # KL1

   $east->element($indexes[0])->add( (-0.268153*0.6)*$KL1 );

# CSR SEXT CUR


@indexes = $west->indexes("^(xv?)"); # KL2

$flag = 1;
for($i = 0; $i < @indexes; $i++){
    $flag *= -1;
    if($flag < 0){ $west->element($indexes[$i])->add( (0.73076*0.272/2.)*$KL2 ); }
    else{ $west->element($indexes[$i])->add( (-0.86092*0.272/2.)*$KL2 );}
}

@indexes = $east->indexes("^(xv?)"); # KL2

$flag = 1;
$esize = @indexes; 
for($i = 0; $i < $esize; $i++){
    $flag *= -1;
    if($flag < 0){ $east->element($indexes[$esize - $i - 1])->add( (0.73076*0.272/2.)*$KL2 ); }
    else{ $east->element($indexes[$esize - $i - 1])->add( (-0.86092*0.272/2.)*$KL2 );}
}

# CSR OCTU CUR

@indexes = $west->indexes("^(oct)"); # KL3

# CSR SQEWQUAD

@west_skql = ( # from Teapot file
     0.1727*0.0,                    # skq2
     0.1727*0.0240463,              # skqv2
     0.1727*0.0,                    # skqv3
     0.1727*0.0,                    # skqv4
     0.1727*0.0,                    # skqv5
     0.1524*0.0,                    # skq7  is not in Teapot file
     0.4075*0.0,                    # skq14 is not in Teapot file
     0.4080*0.0,                    # skq29 is not in Teapot file
     0.1280*0.0,                    # skq47
     0.2540*0.0                     # skq48
);

@indexes = $west->indexes("^(skq[0-9_]+|skqv[0-9_]+)"); # KTL1

for($i = 0; $i < @indexes; $i++){
    $west->element($indexes[$i])->add( ($west_skql[$i])*$KTL1 );
}

@indexes = $east->indexes("^(skq[0-9_]+|skqv[0-9_]+)"); # KTL1

$esize = @indexes;
for($i = 0; $i < $esize; $i++){
    $east->element($indexes[$esize - $i - 1])->add( (-$west_skql[$i])*$KTL1 );
}

# CSR SQEWSEXT

@indexes = $west->indexes("^(skx)"); # KTL2

# CSR VSP VOLT

@indexes = $west->indexes("^(spv)");

#for($i = 0; $i < @indexes; $i++){
#    print $i, " " , $west->element($indexes[$i])->genName, "\n";
#}

1;

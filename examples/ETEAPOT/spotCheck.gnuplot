set format y "%12.7f"

set term x11 0
p 'JT_ONE.deviationFromOrbit' u 2  , 'JT_TWO.deviationFromOrbit' u 2
set term x11 1
p 'JT_THREE.deviationFromOrbit' u 2, 'JT_FOUR.deviationFromOrbit' u 2
set term x11 2
p 'JT_FIVE.deviationFromOrbit' u 2 , 'JT_SIX.deviationFromOrbit' u 2
set term x11 3
p 'JT_SEVEN.deviationFromOrbit' u 2 , 'JT_EIGHT.deviationFromOrbit' u 2

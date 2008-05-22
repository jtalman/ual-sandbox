#include "timer.h"

struct timeb	tb1, tb2;


void start_ms() {
	ftime(&tb1);
}

long end_ms() {
	ftime(&tb2);
	return (tb2.time*1000+tb2.millitm) - (tb1.time*1000+tb1.millitm);
}

long get_ms () {
	struct timeb sTime;

	ftime (&sTime);
	return (sTime.time * 1000 + sTime.millitm);
};


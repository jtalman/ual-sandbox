#ifndef __timer_h__
#define __timer_h__

#include <time.h>
#include <sys/timeb.h>
#include <sys/types.h>


void start_ms ();
long end_ms ();
long get_ms ();

#endif

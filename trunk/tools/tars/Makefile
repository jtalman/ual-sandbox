

GPERF_TARGETS =  --bindir=$(UAL)/tools/bin/$(UAL_ARCH) \
		 --infodir=$(UAL)/tools/info \
		 --mandir=$(UAL)/tools/man 
GPERF_FLAGS   =

FFTW_FLAGS   =  --enable-shared --with-gcc --disable-fortran --enable-static=no

FFTW_TARGETS =  --libdir=$(UAL)/tools/lib/$(UAL_ARCH) \
		--includedir=$(UAL)/tools/include \
		--infodir=.
FFTW_FLAGS   =  --enable-shared --with-gcc --disable-fortran --enable-static=no

PCRE_TARGETS =  --libdir=$(UAL)/tools/lib/$(UAL_ARCH) \
		--includedir=$(UAL)/tools/include \
		--bindir=$(UAL)/tools/bin/$(UAL_ARCH) \
		--mandir=$(UAL)/tools/man \
		--sharedstatedir=$(UAL)/tools/man \
		--docdir=$(UAL)/tools/man

PCRE_FLAGS   =

XML2_TARGETS = 	--bindir=$(UAL)/tools/bin/$(UAL_ARCH) \
		--libdir=$(UAL)/tools/lib/$(UAL_ARCH) \
		--includedir=$(UAL)/tools/include \
		--mandir=$(UAL)/tools/man \
		--datadir=$(UAL)/tools/share 
XML2_FLAGS   = 

GSL_TARGETS = 	--bindir=$(UAL)/tools/bin/$(UAL_ARCH) \
		--libdir=$(UAL)/tools/lib/$(UAL_ARCH) \
		--includedir=$(UAL)/tools/include \
		--mandir=$(UAL)/tools/man \
		--datadir=$(UAL)/tools/share \
		--infodir=$(UAL)/tools/info
GSL_FLAGS   = 

XERCESC_TARGETS = 
XERCESC_FLAGS   = -plinux -cgcc -xg++ -minmem -nsocket -tnative -rpthread

MUPARSER_TARGETS = --prefix=$(UAL)/tools \
		--libdir=$(UAL)/tools/lib/$(UAL_ARCH)


compile:
	gzip -d -c muparser_v134.tar.gz | tar xvf -
	(cd muparser_v134; \
	./configure $(MUPARSER_TARGETS); \
	make; \
	make install; ) 
	rm -rf muparser_v134
	gzip -d -c xerces-c-src_2_7_0.tar.gz | tar xvf -
	(cd xerces-c-src_2_7_0/src/xercesc; \
	autoconf; \
	./runConfigure $(XERCESC_FLAGS); \
	make; \
	cd ../../..; \
	cp  xerces-c-src_2_7_0/lib/* $(UAL)/tools/lib/$(UAL_ARCH); \
	cp -r xerces-c-src_2_7_0/include/xercesc $(UAL)/tools/include; )
	rm -rf xerces-c-src_2_7_0
	gzip -d -c gperf-2.7.2.tar.gz | tar xvf -
	(cd gperf-2.7.2; \
	./configure  $(GPERF_TARGETS) $(GPERF_FLAGS); \
	make; \
	make install;) 
	rm -rf gperf-2.7.2
	gzip -d -c pcre-8.02.tar.gz | tar xvf -
	(cd pcre-8.02; \
	./configure  $(PCRE_TARGETS) $(PCRE_FLAGS); \
	make; \
	make install;) 
	rm -rf pcre-8.02
	gzip -d -c libxml2-sources-2.6.22.tar.gz | tar xvf -
	(cd libxml2-2.6.22; \
	./configure  $(XML2_TARGETS) $(XML2_FLAGS); \
	mv Makefile Makefile.orig; \
	cp Makefile.orig Makefile; \
	make; \
	make install;) 
	rm -rf libxml2-2.6.22
	gzip -d -c fftw-2.1.3.nm.tar.gz | tar xvf -
	(cd fftw-2.1.3.nm; \
	./configure  $(FFTW_TARGETS) $(FFTW_FLAGS); \
	make; \
	make install;) 
	rm -rf fftw-2.1.3.nm
	gzip -d -c gsl-1.4.tar.gz | tar xvf -
	(cd gsl-1.4; \
	./configure  $(GSL_TARGETS) $(GSL_FLAGS); \
	make; \
	make install;) 
	rm -rf gsl-1.4
clean:
	rm -rf  $(UAL)/tools/lib/$(UAL_ARCH)/* \
		$(UAL)/tools/include/* \
		$(UAL)/tools/bin/$(UAL_ARCH)/* \
		$(UAL)/tools/man/* \
		$(UAL)/tools/share/*


CONFIG = ../tools.config
include ${CONFIG}

all: kwarg argweaver relate relate_lib tsinfer

kwarg: 
	git clone ${KWARG_GITREPO}
	cd kwarg && \
	  git checkout tags/${KWARG_VERSION} && \
	  cd source && \
	  make

argweaver: 
	git clone ${ARGWEAVER_GITREPO}
	# checkout a specific revision
	cd argweaver && \
	  git checkout ${ARGWEAVER_VERSION} && \
	  make && \
	  python2 setup.py install --user  # annoyingly python2 is required for smc2arg

relate: 
	git clone ${RELATE_GITREPO}
	# checkout a specific revision (latest as of Mar 2022)
	# NB: removes gzstream version file which causes recent compilations to fail
	cd relate && \
	  git checkout ${RELATE_VERSION} && \
	  rm -f include/src/gzstream/version && \
	  mkdir -p build && \
	  cd build && \
	  cmake .. && \
	  make

relate_lib: 
	git clone ${RELATELIB_GITREPO}
	# checkout a specific revision (latest dev as of Mar 2022)
	# NB: removes gzstream version file which causes recent compilations to fail
	cd relate_lib && \
	  git checkout ${RELATELIB_VERSION} && \
	  rm -f include/src/gzstream/version && \
	  mkdir -p build && \
	  cd build && \
	  cmake .. && \
	  make

tsinfer: 
	git clone ${TSINFER_GITREPO}
	# checkout a specific revision (latest dev as of Mar 2022)
	cd tsinfer && \
	  git checkout tags/${TSINFER_VERSION} && \
	  make

clean:
	rm -rf kwarg
	rm -rf argweaver
	rm -rf relate
	rm -rf relate_lib
	rm -rf tsinfer
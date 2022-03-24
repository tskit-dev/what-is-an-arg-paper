all: argweaver relate relate_lib

argweaver: 
	git clone https://github.com/CshlSiepelLab/argweaver
	# checkout a specific revision (latest as of Mar 2022)
	cd argweaver && \
	  git checkout 7191343c1d43c02de04e4e0a150d77b0612728c6 && \
	  make && \
	  python2 setup.py install --user

relate: 
	git clone https://github.com/MyersGroup/relate
	# checkout a specific revision (latest as of Mar 2022)
	cd relate && \
	  git checkout eea4a89b1d5eba5561f64d2a96eb64aaab6cd91c && \
	  mkdir build && \
	  cd build && \
	  cmake .. && \
	  make

relate_lib: 
	git clone https://github.com/leospeidel/relate_lib
	# checkout a specific revision (latest dev as of Mar 2022)
	cd relate_lib && \
	  git checkout 0b461f3c7282d784e5a65122c029bf4a857cbee5 && \
	  mkdir build && \
	  cd build && \
	  cmake .. && \
	  make

clean:
	rm -rf argweaver
	rm -rf relate
	rm -rf relate_lib
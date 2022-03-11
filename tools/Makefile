all: argweaver

argweaver: 
	git clone https://github.com/CshlSiepelLab/argweaver
	# checkout a specific revision (latest as of Mar 2022)
	cd argweaver && git checkout 7191343c1d43c02de04e4e0a150d77b0612728c6 && make && python2 setup.py install --user

clean:
	rm -rf argweaver
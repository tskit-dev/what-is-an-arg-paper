DATA=
FIGURES=
# To add a new illustration, add a line for the corresponding figure name
# here (use hyphens to separate words, not underscores). Then, define
# the corresponding function/command in illustrations.py, making
# sure that it writes out the corresponding SVG file. The Makefile will
# take care of the rest.
ILLUSTRATIONS=\
	illustrations/ancestry-resolution.pdf \
	illustrations/simplification.pdf \
	illustrations/arg-in-pedigree.pdf \

all: paper.pdf

paper.pdf: paper.tex paper.bib ${DATA} ${FIGURES} ${ILLUSTRATIONS}
	pdflatex -shell-escape paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex

illustrations/%.svg: illustrations.py
	python3 illustrations.py $*

%.pdf : %.svg
	# Needs inkscape >= 1.0
	inkscape --export-type="pdf" $<

paper.ps: paper.dvi
	dvips paper

paper.dvi: paper.tex paper.bib
	latex paper.tex
	bibtex paper
	latex paper.tex
	latex paper.tex

.PHONY: spellcheck
spellcheck: aspell.conf
	aspell --conf ./aspell.conf --check paper.tex

clean:
	rm -f *.pdf
	rm -f illustrations/*.pdf
	rm -f illustrations/*.svg
	rm -f *.log *.dvi *.aux
	rm -f *.blg *.bbl

mrproper: clean
	rm -f *.ps *.pdf

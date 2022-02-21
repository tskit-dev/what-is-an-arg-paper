DATA=
FIGURES=
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

illustrations/ARG_edge_annotations.svg: illustrations.py
	python3 illustrations.py arg-edge-annotations


%.pdf : %.svg
	# For inkscape >= 1.0
	inkscape --export-type="pdf" $<
	# For inkscape < 1, this works (but is missing some shading)
	# inkscape $< --export-pdf=$@
	# This gives a faithful conversion to pdf, but needs some page trimming
	# chromium --headless --no-margins --print-to-pdf=$@ $<

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


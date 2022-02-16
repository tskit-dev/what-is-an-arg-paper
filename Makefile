DATA=
FIGURES=
ILLUSTRATIONS=\
	illustrations/ARG_edge_annotations.pdf \
	illustrations/ARG_recomb_node_deletion.pdf \
	illustrations/pedigree_figure.pdf \


all: paper.pdf

paper.pdf: paper.tex paper.bib ${DATA} ${FIGURES} ${ILLUSTRATIONS}
	pdflatex -shell-escape paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex

illustrations/ARG_recomb_node_deletion.svg: illustrations/ARG_recomb_node_deletion.py
	python3 $<

illustrations/ARG_edge_annotations.svg: illustrations/ARG_edge_annotations.py
	python3 $<

illustrations/pedigree_figure.svg: illustrations/pedigree.py
	python3 $<

# NB not reflected in this makefile running pedigree.py also creates pedigree_ARG.pdf

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
	rm -f *.log *.dvi *.aux
	rm -f *.blg *.bbl
	rm -f *.eps *.[1-9]
	rm -f src/*.mpx *.mpx

mrproper: clean
	rm -f *.ps *.pdf


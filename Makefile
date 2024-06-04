DATA=
FIGURES=
# To add a new illustration, add a line for the corresponding figure name
# here (use hyphens to separate words, not underscores). Then, define
# the corresponding function/command in illustrations.py, making
# sure that it writes out the corresponding SVG file. The Makefile will
# take care of the rest.
# NB: ensure that any additional static files that must be stored in
# git are put in the illustrations/assets directory.
ILLUSTRATIONS=\
	illustrations/ancestry-resolution.pdf \
	illustrations/simplification.pdf \
	illustrations/arg-in-pedigree.pdf \
	illustrations/inference.pdf \
	illustrations/cell-lines.pdf \
	illustrations/simplification-with-edges.pdf \

all: paper.pdf response-to-reviewers.pdf response-to-reviewers-2.pdf

paper.pdf: paper.tex paper.bib ${DATA} ${FIGURES} ${ILLUSTRATIONS}
	pdflatex -shell-escape paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex

inference_inputs: \
	examples/Kreitman_SNP.samples \
	examples/Kreitman_SNP.matrix \
	examples/Kreitman_SNP.sites \
	examples/Kreitman_SNP.haps \

inference_outputs: \
	inference_inputs \
	examples/Kreitman_SNP_tsinfer.trees \
	examples/Kreitman_SNP_kwarg.trees \
	examples/Kreitman_SNP_argweaver.trees \
	examples/Kreitman_SNP_relate_merged.trees \

examples/Kreitman_SNP.samples: make_example_inputs.py
	python3 	make_example_inputs.py tsinfer-input

examples/Kreitman_SNP.matrix: make_example_inputs.py
	python3 	make_example_inputs.py kwarg-input

examples/Kreitman_SNP.sites: make_example_inputs.py
	python3 	make_example_inputs.py argweaver-input

examples/Kreitman_SNP.haps: make_example_inputs.py
	python3 	make_example_inputs.py relate-input

examples/Kreitman_SNP_tsinfer.trees: make_example_outputs.py
	python3 	make_example_outputs.py run-tsinfer

examples/Kreitman_SNP_kwarg.trees: make_example_outputs.py
	python3 	make_example_outputs.py run-kwarg

examples/Kreitman_SNP_argweaver.trees: make_example_outputs.py
	python3 	make_example_outputs.py run-argweaver

examples/Kreitman_SNP_relate_merged.trees: make_example_outputs.py
	python3 	make_example_outputs.py run-relate

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
	rm -f *.log *.dvi *.aux
	rm -f *.blg *.bbl

mrproper: clean
	rm -f illustrations/*.pdf
	rm -f illustrations/*.svg
	rm -f *.ps *.pdf


review-diff.tex: paper.tex
	latexdiff reviewed-paper.tex paper.tex > review-diff.tex

review-diff.pdf: review-diff.tex
	pdflatex review-diff.tex
	pdflatex review-diff.tex
	bibtex review-diff
	pdflatex review-diff.tex

response-to-reviewers.pdf: response-to-reviewers.tex
	pdflatex $<

review-diff-2.tex: paper.tex
	latexdiff reviewed-paper-2.tex paper.tex > review-diff-2.tex

review-diff-2.pdf: review-diff-2.tex
	pdflatex review-diff-2.tex
	pdflatex review-diff-2.tex
	bibtex review-diff-2
	pdflatex review-diff-2.tex

response-to-reviewers-2.pdf: response-to-reviewers-2.tex
	pdflatex $<

TARGETS = Project_Report_EE107_LC_AN

# Outputs [i.e. TARGET.pdf]
PDFS = $(addsuffix .pdf,$(TARGETS))

# Inputs/Dependancies
COMMON_TEX = $(filter-out $(addsuffix .tex,$(TARGETS)),$(wildcard *.tex))
REFERENCES = references

IMAGE_DIR=../images
IMAGES = $(wildcard $(IMAGE_DIR)/*.png) $(wildcard $(IMAGE_DIR)/*.pdf) $(PS2PDFS)
DEPENDANCIES = $(COMMON_TEX) $(IMAGES)


all: $(PDFS)

%.pdf: %.tex %.bbl $(DEPENDANCIES)
	pdflatex $*
	pdflatex $*

%.bbl: %.tex ${REFERENCES}.bib $(PS2PDFS) $(DEPENDANCIES)
	-pdflatex $*
	bibtex $*.aux

clean:
	rm -f *.toc *.aux *.log *.out

nuke:
	rm -f *.toc *.pdf *.dvi *.aux *.log *.out *.bbl *.blg

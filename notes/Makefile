#
# Makefile
# Clay L. McLeod, 2016-01-23 20:55
#

default: pdf ps clean

pdf:
	@cd ./src; pdflatex notes.tex; bibtex notes; pdflatex notes; pdflatex notes; mv notes.pdf ../out/

ps:
	@cd ./src; latex notes; bibtex notes; latex notes; latex notes; dvips notes; mv notes.ps ../out/

.PHONY: clean
clean:
	@cd ./src && ls | grep -v notes.tex | grep -v notes.bib | xargs rm 

# vim:ft=make
#

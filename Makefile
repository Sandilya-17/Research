.PHONY: install test quick full report paper clean
install:
	pip install -r requirements.txt
test:
	python -m pytest tests -q
quick:            # ~1 min smoke test
	python scripts/run_all.py --quick
full:             # ~5 min on one CPU core
	python scripts/run_all.py
report:
	python scripts/make_report.py
paper: report
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex
clean:
	rm -rf data data_quick results_quick paper/*.aux paper/*.log

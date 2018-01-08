.PHONY: training submission default

default: training

training:
	python3 -m learn.train

submission: 
	zip -r submission.zip MyBot.py LANGUAGE hlt/


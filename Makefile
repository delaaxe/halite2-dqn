.PHONY: training submission default

default: training

training:
	python3 -m dqn.learn

submission:
	rm -f submission.zip
	zip -r submission.zip MyBot.py LANGUAGE dqn_model.pkl install.sh hlt/ dqn/


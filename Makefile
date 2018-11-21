data/train:
	curl -o 'uci.zip' 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
	unzip uci.zip
	mv UCI\ HAR\ Dataset/* data/
	rm -rf UCI\ HAR\ Dataset/
	rm -rf __MACOSX/
	rm uci.zip

AUDIO ?= data/peggysue.mp3
TEXT ?= data/peggysue.txt
BATCH_SIZE ?= 1
MODEL_ID ?= facebook/wav2vec2-large-960h-lv60-self
DEVICE ?= mps

install:
	pip install -e .
build:
	uv build 
test:
	pytest 
run:
	lyricaligner \
	--audio $(AUDIO) \
	--text $(TEXT) \
	--device $(DEVICE) \
	--batch_size $(BATCH_SIZE) \
	--model_id $(MODEL_ID) 

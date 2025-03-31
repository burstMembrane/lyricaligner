AUDIO ?= data/peggysue.mp3
TEXT ?= data/peggysue.txt

test:
	pytest 
run:
	lyricaligner \
	--audio $(AUDIO) \
	--text $(TEXT) \
	--device mps \
	--batch_size 1 \
	--model_id sicto/wav2vec2-large-960h-alt 
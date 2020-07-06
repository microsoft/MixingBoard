# DialoGPT forward and reverse model 
mkdir models/DialoGPT
wget https://convaisharables.blob.core.windows.net/lsp/multiref/medium_ft.pkl -O models/DialoGPT/medium_ft.pkl
wget https://convaisharables.blob.core.windows.net/lsp/multiref/small_reverse.pkl -O models/DialoGPT/small_reverse.pkl

# BiDAF model 
mkdir models/BiDAF
wget https://storage.googleapis.com/allennlp-public-models/bidaf-model-2020.03.19.tar.gz -O models/BiDAF/bidaf-model-2020.03.19.tar.gz

# Content Transfer
mkdir models/crg
wget http://tts.speech.cs.cmu.edu/content_transfer/crg_model.zip -O temp/crg_model.zip
unzip temp/crg_model.zip -d temp
mv temp/crg_model/crg_model/crg_model.pt models/crg/crg_model.pt

wget http://tts.speech.cs.cmu.edu/content_transfer/sentencepieceModel.zip -O temp/bpe.zip
unzip temp/bpe.zip -d temp
mv temp/bpe/sentencepieceModel/bpeM.vocab models/crg/bpeM.vocab
mv temp/bpe/sentencepieceModel/bpeM.model models/crg/bpeM.model
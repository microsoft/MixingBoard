# install python packages ====

conda install pytorch -c pytorch
pip install azure-cognitiveservices-search-websearch==1.0.0
pip install git+https://github.com/boudinfl/pke.git
pip install allennlp allennlp-models
pip install flask flask_restful
pip install spacy regex nltk pyaudio
pip install sentencepiece sacremoses

python -m nltk.downloader stopwords
python -m nltk.downloader wordnet
python -m nltk.downloader universal_tagset
python -m spacy download en
python -m spacy download en_core_web_sm

# download an older version of Transformers that are compatible to the current MixingBoard
mkdir temp
wget https://github.com/huggingface/transformers/archive/4d45654.zip -O temp/transformers.zip
tar -xf temp/transformers.zip -C temp
move temp/transformers-4d456542e9d381090f9a00b2bcc5a4cb07f6f3f7/transformers src/transformers
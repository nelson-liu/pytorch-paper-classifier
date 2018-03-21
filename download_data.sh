mkdir -p data
cd data

# Download train data
wget -N https://s3-us-west-2.amazonaws.com/allennlp/datasets/academic-papers-example/train.jsonl

# Download validation data
wget -N https://s3-us-west-2.amazonaws.com/allennlp/datasets/academic-papers-example/dev.jsonl

# Download GloVe vectors
wget -N https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz

cd ..

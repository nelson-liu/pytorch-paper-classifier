mkdir -p data
cd data

# Download train data
wget https://s3-us-west-2.amazonaws.com/allennlp/datasets/academic-papers-example/train.jsonl

# Download validation data
wget https://s3-us-west-2.amazonaws.com/allennlp/datasets/academic-papers-example/dev.jsonl

cd ..

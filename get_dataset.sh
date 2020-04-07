#!/bin/bash

mkdir dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wJXbKPE8-LNMeCi6SOoZ0sPyqbJFfidY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wJXbKPE8-LNMeCi6SOoZ0sPyqbJFfidY" -O dataset.tar && rm -rf /tmp/cookies.txt
unzip jpg.zip
mv jpg/ dataset/jpg/
rm jpg.zip

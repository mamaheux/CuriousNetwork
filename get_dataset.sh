#!/bin/bash

mkdir dataset
wget https://curiousnetworkdatabase.blob.core.windows.net/database/jpg.zip
unzip jpg.zip
mv jpg/ dataset/jpg/
rm jpg.zip

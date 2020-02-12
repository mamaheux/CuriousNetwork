#!/bin/bash

mkdir database
wget https://curiousnetworkdatabase.blob.core.windows.net/database/jpg.zip
unzip jpg.zip
mv jpg/ database/jpg/
rm jpg.zip

#!/bin/sh

CLFWZIP="lfwcrop_grey.zip"

if [ -f $CLFWZIP ]; then
  rm $CLFWZIP
fi
wget http://itee.uq.edu.au/~conrad/lfwcrop/lfwcrop_grey.zip
unzip $CLFWZIP
rm $CLFWZIP

MLTAR="ml-data.tar__0.gz"
wget http://www.grouplens.org/system/files/ml-data.tar__0.gz
tar xvfz $MLTAR
rm $MLTAR

echo "Done!"
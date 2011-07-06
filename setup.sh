#!/bin/sh

CLFWZIP="lfwcrop_grey.zip"

if [ -f $CLFWZIP ]; then
  rm $CLFWZIP
fi
wget http://itee.uq.edu.au/~conrad/lfwcrop/lfwcrop_grey.zip
unzip $CLFWZIP
rm $CLFWZIP

echo "Done!"
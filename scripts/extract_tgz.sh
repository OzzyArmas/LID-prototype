#!usr/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: $0 <tgz file> <language>"
  exit 1
fi

ZIP=$1
LANG=$2
TARGET="/tmp/data/voxforge/wav"/$LANG
TEMP_DIR=$(dirname ${ZIP})
FILE_NAME=$(basename ${ZIP})
STRIPPED_FILE_NAME=${FILE_NAME%.tgz}
STRIPPED_FILE_NAME=${STRIPPED_FILE_NAME}

tar -xf $ZIP -C $TEMP_DIR

UNZIPPED_FOLDER=$TEMP_DIR/${STRIPPED_FILE_NAME}
WAVES=$UNZIPPED_FOLDER/wav
  
if [ ! -d $TARGET ]; then
  mkdir -p $TARGET
fi

for WAVE in $(ls ./$WAVES)
do
  mv $WAVES/$WAVE $TARGET/$STRIPPED_FILE_NAME-$WAVE
done

echo rm -f $ZIP
echo rm -rf $UNZIPPED_FOLDER

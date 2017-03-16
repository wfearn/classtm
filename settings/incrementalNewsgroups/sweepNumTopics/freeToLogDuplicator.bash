#!/usr/bin/env bash

for file in *free.settings;
do
  cp $file ${file/free/tsvm};
  sed -i 's/quickincfree/inctsvm/' ${file/free/tsvm}
  sed -i 's/free/tsvm/' ${file/free/tsvm}
done

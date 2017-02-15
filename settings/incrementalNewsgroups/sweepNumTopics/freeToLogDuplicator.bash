#!/usr/bin/env bash

for file in *free.settings;
do
  cp $file ${file/free/log};
  sed -i 's/quickincfree/inclog/' ${file/free/log}
  sed -i 's/free/log/' ${file/free/log}
done

#!/usr/bin/env bash

for file in *free.settings;
do
  cp $file ${file/free/nb};
  sed -i 's/quickincfree/incnb/' ${file/free/nb}
  sed -i 's/free/nb/' ${file/free/nb}
done

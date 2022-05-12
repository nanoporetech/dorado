#! /usr/bin/env bash

rm -rf koi_lib
KOI_REV=$(git submodule status | grep koi | awk {'print substr($1,1,8)'})
echo "Koi revision is ${KOI_REV}"
mkdir koi_lib
mkdir koi_lib/lib
cp ./build/libkoi*.a koi_lib/lib
mkdir koi_lib/include
cp ./dorado/3rdparty/koi/koi/lib/*.h koi_lib/include
tar -cvzf koi_lib_linux_${KOI_REV}.tar.gz ./koi_lib/
rm -rf koi_lib
echo "done"
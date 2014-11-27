#!/bin/bash

echo "Comping scripts to $1"
shopt -u dotglob
rm -rf $1/scripts
mkdir $1/scripts
cp -r scripts/* $1/scripts
chmod -R +x $1/scripts

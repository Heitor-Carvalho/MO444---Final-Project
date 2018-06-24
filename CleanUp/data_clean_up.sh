#!/bin/bash

CLEANUP_FILE=$1

echo "Expanding abreviation"
sed -i $CLEANUP_FILE -e "s/'s/ is/g" -e "s/'t/ it/g" -e "s/'m/ am/g" -e "s/'ve/ have/g" -e "s/'ll/ will/g" -e "s/'d/ would/g" -e "s/'re/ are/g"

echo "Replacing text numbers"
sed -i $CLEANUP_FILE -e "s/ one / 1 /g" -e "s/ two / 2 /g" -e "s/ three / 3 /g" -e "s/ four / 4 /g" -e "s/ five / 5 /g" -e "s/ six / 6 /g" -e "s/ seven / 7 /g" -e "s/ eigth / 8 /g" -e "s/ nine / 9 /g" -e "s/ ten / 10 /g"

echo "Cleaning Non ASCII character"
perl -i.bak -pe 's/[^[:ascii:]]//g' $CLEANUP_FILE

echo "Here there are you new training set"

#!/bin/sh

wparam=10
codedir=`dirname $0`/../code #adaptable
#codedir='~/ap/code' #robust

for fn
do
bn=`basename "$fn"`
echo "$codedir/ap.py $wparam < \"$fn\""
done

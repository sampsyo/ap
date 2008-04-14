#!/bin/sh

wparam=2
codedir=`dirname $0`/../code #adaptable
# codedir='~/ap/code' #robust
rundir='~/ap/testing'

for fn
do
bn=`basename "$fn"`
echo "cd $rundir; $codedir/ap.py $wparam < \"$fn\""
done

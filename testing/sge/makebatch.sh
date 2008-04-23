#!/bin/sh

wparam=10
codedir=`dirname $0`/../code #adaptable
# codedir='~/ap/code' #robust
rundir='~/ap/testing'

for fn
do
bn=`basename "$fn"`
echo "qsub ap/testing/sge/apjob.sh -- $bn $wparam"
done

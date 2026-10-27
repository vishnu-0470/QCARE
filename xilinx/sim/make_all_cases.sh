#!/bin/bash
cd "$(dirname "$0")/../.."
octave-cli --quiet --eval "REC=100;T0=0;DUR=120;TAG='rA';SEED=3;GAIN=38;NOISE=0;ART=0;TAU=100;source('matlab/sim/make_vectors.m')" >/dev/null 2>&1
octave-cli --quiet --eval "REC=119;T0=300;DUR=150;TAG='rB';SEED=5;GAIN=38;NOISE=8;ART=1;TAU=0;source('matlab/sim/make_vectors.m')" >/dev/null 2>&1
octave-cli --quiet --eval "REC=119;T0=100;DUR=150;TAG='rC';SEED=9;GAIN=45;NOISE=3;ART=0;TAU=256;source('matlab/sim/make_vectors.m')" >/dev/null 2>&1
octave-cli --quiet --eval "REC=100;T0=600;DUR=100;TAG='rD';SEED=21;GAIN=30;NOISE=15;ART=1;TAU=180;source('matlab/sim/make_vectors.m')" >/dev/null 2>&1
octave-cli --quiet --eval "REC=119;T0=0;DUR=300;TAG='rE';SEED=13;GAIN=38;NOISE=5;ART=1;TAU=150;source('matlab/sim/make_vectors.m')" >/dev/null 2>&1
echo VECTORS-DONE

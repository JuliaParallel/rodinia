#!/usr/bin/env python
# encoding: utf-8
"""
Created by Cole Trapnell and Michael Schatz.
"""

import sys
import getopt
import random


help_message = """
genreads outputs a multi-FASTA file containing a random sampling of
read-sized subsequences of the provided reference sequence.

Usage:
genreads [options] <reference file> <length of reads> <# of reads>

Options:
-s <integer value> --seed=        the seed for the random number
                                  generator
                                  """


                                  class Usage(Exception):
                                      def __init__(self, msg):
self.msg = msg


def main(argv=None):

    if argv is None:
argv = sys.argv
try:
try:
    opts, args = getopt.getopt(argv[1:], "hs:v", ["help", "seed="])
except getopt.error, msg:
    raise Usage(msg)

seed_val = 0
# option processing
for option, value in opts:
    if option == "-v":
        verbose = True
    if option in ("-h", "--help"):
        raise Usage(help_message)
    if option in ("-s", "--seed"):
        seed_val = long(value)

random.seed(seed_val)

if len(args) != 3:
    raise Usage(help_message)

fasta_input = args[0]
read_length = int(args[1])
num_reads = int(args[2])

f = open(fasta_input, "r")
lines = f.readlines()

if lines[0].find(">") == -1:
    raise Usage("File is not FASTA format")

seq = "".join([line.strip() for line in lines[1:]])

L = len(seq)

rid = 0

for i in range(0, num_reads):
    start = random.randint(0, L - read_length)
    end = start + read_length
    rid += 1
    print ">rid" + str(rid) + " " + str(start+1) + "-" + str(end+1)
    print seq[start:end]

    except Usage, err:
print >> sys.stderr, sys.argv[0].split("/")[-1] + ": " + str(err.msg)
return 2

if __name__ == "__main__":
    sys.exit(main())

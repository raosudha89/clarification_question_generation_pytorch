#!/usr/bin/perl -w
use strict;

my $usage = "usage: cat FILE | all_ngrams.pl [-noboundary|-smallboundary] <N>\n";

my $boundary = 2;
my $N = 1;

while (1) {
    my $tmp = shift or die $usage;
    if ($tmp eq '-noboundary') { $boundary = 0; }
    elsif ($tmp eq '-smallboundary') { $boundary = 1; }
    elsif ($tmp =~ /^[0-9]+$/) { $N = $tmp; last; }
    else { die $usage; }
}

while (<>) {
    chomp;
    if (/^[\s]*$/) { next; }
    my @w = split;
    my $M = scalar @w;

    my $lo = -$N+1;
    if ($boundary == 0) { $lo = 0; }
    if (($boundary == 1) && ($N>1)) { $lo = -1; }

    my $hi = $M;
    if ($boundary == 0) { $hi = $M-$N+1; }
    if (($boundary == 1) && ($N>1)) { $hi = $M-$N+2; }

    for (my $i=$lo; $i<$hi; $i++) {
        for (my $j=0; $j<$N; $j++) {
            if ($j > 0) { print ' '; }
            print (($i+$j<0) ? '<s>' : (($i+$j>=$M) ? '</s>' : $w[$i+$j]));
        }
        print "\n";
    }
}

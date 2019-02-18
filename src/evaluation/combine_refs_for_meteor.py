import sys
import argparse

def main(args):
    ref_sents = [None]*int(args.no_of_refs)
    for i in range(int(args.no_of_refs)):
        with open(args.ref_prefix+str(i), 'r') as f:
            ref_sents[i] = [line.strip('\n') for line in f.readlines()]
    
    combined_ref_file = open(args.combined_ref_fname, 'w')
    for i in range(len(ref_sents[0])):
        for j in range(int(args.no_of_refs)):
            combined_ref_file.write(ref_sents[j][i]+'\n')
    combined_ref_file.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--ref_prefix", type = str)
    argparser.add_argument("--no_of_refs", type = int)
    argparser.add_argument("--combined_ref_fname", type = str)
    args = argparser.parse_args()
    print args
    print ""
    main(args)

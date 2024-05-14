from transformers import pipeline
import logging, os, csv, distance

fill_mask = pipeline('fill-mask', model='distilbert-base-uncased')
mask = fill_mask.tokenizer.mask_token

def get_typo_locations(fh):
    tsv_f = csv.reader(fh, delimiter='\t')
    for line in tsv_f:
        yield (
            # line[0] contains the comma separated indices of typo words
            [int(i) for i in line[0].split(',')],
            # line[1] contains the space separated tokens of the sentence
            line[1].split()
        )
        
def new_score(typo, prediction, lamba=1):
    levenshtein_dist = distance.levenshtein(typo, prediction['token_str'])
    score = (1 - lamba) * prediction['score'] + lamba * (1 / (levenshtein_dist + 1))
    #score = prediction['score']/(levenshtein_dist + 1)
    return score

def select_correction(typo, predict):
    # return the most likely prediction for the mask token
    score = [new_score(typo, p) for p in predict]
    max_score_index = score.index(max(score))   
    return predict[max_score_index]['token_str']
    # distances = [distance.levenshtein(typo, p['token_str']) for p in predict]
    # #print(distances)
    # #print([p['token_str'] for p in predict])
    # min_distance_index = distances.index(min(distances))
    # return predict[min_distance_index]['token_str']

def spellchk(fh):
    for (locations, sent) in get_typo_locations(fh):
        spellchk_sent = sent
        for i in locations:
            # predict top_k replacements only for the typo word at index i
            predict = fill_mask(
                " ".join([ sent[j] if j != i else mask for j in range(len(sent)) ]), 
                top_k=500
            )
            logging.info(predict)
            spellchk_sent[i] = select_correction(sent[i], predict)
        yield(locations, spellchk_sent)

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfile", 
                            dest="input", 
                            default=os.path.join('data', 'input', 'dev.tsv'), 
                            help="file to segment")
    argparser.add_argument("-l", "--logfile", 
                            dest="logfile", 
                            default=None, 
                            help="log file for debugging")
    opts = argparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    with open(opts.input) as f:
        for (locations, spellchk_sent) in spellchk(f):
            print("{locs}\t{sent}".format(
                locs=",".join([str(i) for i in locations]),
                sent=" ".join(spellchk_sent)
            ))

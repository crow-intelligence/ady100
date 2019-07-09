import operator
import itertools
from collections import Counter
from os.path import join

import nltk

out_path = 'data/processed/ners'

wd_ner = []
with open('data/interim/ner/ady.out', 'r') as f:
    for l in f:
        l = l.strip().split('\t')
        if len(l) == 2:
            wd, tag = l[0], l[1]
            wd_ner.append((wd, tag))


def copy_case(str1, str2):
    if str1.isupper():
        return str2.upper()
    elif str1.islower():
        return str2.lower()
    elif str1.istitle():
        return str2.title()
    elif str1.isalpha():
        return copy_case(str1[0], str2[0]) + copy_case(str1[1:], str2[1:])
    else:
        return str2

wd_stem = {}
with open('data/interim/ml/ady.out', 'r') as f:
    for l in f:
        l = l.strip().split('\t')
        if len(l) == 4:
            wd = l[0].title()
            stem = l[1].title()
            wd_stem[wd] = stem


def stem_ner(ner):
    ner = ner.strip().split()
    ner = [e.strip().title() for e in ner]
    if len(ner) == 1:
        if ner[0] in wd_stem:
            return wd_stem[ner[0]]
        else:
            return ner[0]
    else:
        first_part = ' '.join(ner[:-1])
        if ner[-1] in wd_stem:
            stemmed_last = wd_stem[ner[-1]]
        else:
            stemmed_last = ner[-1]
        stemmed_ner = first_part + ' ' + stemmed_last
        return stemmed_ner


person_pattern = r'KT: {<I-PER>+}'
org_pattern = r'KT: {<I-ORG>+}'
loc_pattern = r'KT: {<I-LOC>+}'
misc_pattern = r'KT: {<I-MISC>+}'

person_chunker = nltk.chunk.regexp.RegexpParser(person_pattern)
person_chunks = nltk.chunk.tree2conlltags(person_chunker.parse(wd_ner))
persons = [' '.join(word for word, pos, chunk in group)
           for key, group in
           itertools.groupby(person_chunks,
                             lambda e: e[1] == 'I-PER')
           if key]
persons = [stem_ner(e) for e in persons]
persons = Counter(persons)
sorted_persons = sorted(persons.items(),
                        key=operator.itemgetter(1),
                        reverse=True)
with open(join(out_path, 'persons.tsv'), 'w') as f:
    h = 'Entity\tFrequency\n'
    f.write(h)
    for e in sorted_persons:
        o = e[0] + '\t' + str(e[1]) + '\n'
        f.write(o)

location_chunker = nltk.chunk.regexp.RegexpParser(location_pattern)
location_chunks = nltk.chunk.tree2conlltags(location_chunker.parse(wd_ner))
locations = [' '.join(word for word, pos, chunk in group)
           for key, group in
           itertools.groupby(location_chunks,
                             lambda e: e[1] == 'I-LOC')
           if key]
locations = [stem_ner(e) for e in locations]
locations = Counter(locations)
sorted_locations = sorted(locations.items(),
                        key=operator.itemgetter(1),
                        reverse=True)
with open(join(out_path, 'locations.tsv'), 'w') as f:
    h = 'Entity\tFrequency\n'
    f.write(h)
    for e in sorted_locations:
        o = e[0] + '\t' + str(e[1]) + '\n'
        f.write(o)

organization_chunker = nltk.chunk.regexp.RegexpParser(organization_pattern)
organization_chunks = nltk.chunk.tree2conlltags(organization_chunker.parse(wd_ner))
organizations = [' '.join(word for word, pos, chunk in group)
           for key, group in
           itertools.groupby(organization_chunks,
                             lambda e: e[1] == 'I-ORG')
           if key]
organizations = [stem_ner(e) for e in organizations]
organizations = Counter(organizations)
sorted_organizations = sorted(organizations.items(),
                        key=operator.itemgetter(1),
                        reverse=True)
with open(join(out_path, 'organizations.tsv'), 'w') as f:
    h = 'Entity\tFrequency\n'
    f.write(h)
    for e in sorted_organizations:
        o = e[0] + '\t' + str(e[1]) + '\n'
        f.write(o)

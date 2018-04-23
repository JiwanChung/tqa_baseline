# -*- coding: utf-8 -*-

import sys
import re

import spacy
spacy_en = spacy.load('en')

from revtok import tokenize as revtok_tokenize

class Tokenizer():
    def __init__(self, opt='spacy'):
        if opt == 'spacy':
            self.tokenize = self.spacy
        elif opt == 'revtok':
            self.tokenize = self.revtok
        else:
            self.tokenize = self.revtok

        self.re_dot = re.compile(r".$")
        self.re_que = re.compile(r"\?+")
        self.re_com = re.compile(r"\,+")
        self.re_exc = re.compile(r"\!+")
        self.re_bra = re.compile(r"[ ]+")

        self.re_all = re.compile(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]")

    def re(self, text):
        text = self.re_all.sub(" ", str(text))

        text = self.re_dot.sub(" ", text)
        text = self.re_que.sub("!", text)
        text = self.re_com.sub(",", text)
        text = self.re_exc.sub("?", text)
        text = self.re_bra.sub("", text)

        if sys.version_info < (3, 0):
            text = unicode(text)

        return text

    def spacy(self, text):
        text = self.re(text)

        print_list = [tok.text for tok in spacy_en.tokenizer(text) if tok.text != " "]

        return print_list

    def revtok(self, text):

        text = self.re(text)

        return revtok_tokenize(text)

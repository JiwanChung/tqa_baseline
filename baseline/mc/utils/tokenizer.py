# -*- coding: utf-8 -*-

import re
import six

from revtok import tokenize as revtok_tokenize
import spacy
spacy_en = spacy.load('en')


class Tokenizer():
    def __init__(self, opt='spacy'):
        if opt == 'spacy':
            self.tokenize = self.spacy
        elif opt == 'revtok':
            self.tokenize = self.revtok
        else:
            self.tokenize = self.revtok

        self.re_dot = re.compile(r"\.$")
        self.re_que = re.compile(r"\?+")
        self.re_com = re.compile(r"\,+")
        self.re_exc = re.compile(r"\!+")
        self.re_bra = re.compile(r"[\[\]]+")

    def re(self, text):
        text = str(text)

        text = self.re_dot.sub(" ", text)
        text = self.re_que.sub("!", text)
        text = self.re_com.sub(",", text)
        text = self.re_exc.sub("?", text)
        text = self.re_bra.sub("", text)
        text = text.strip()

        if six.PY2:
            text = unicode(text)

        return text

    def spacy(self, text):
        text = self.re(text)

        print_list = [tok.text for tok in spacy_en.tokenizer(text) if tok.text != " "]

        return print_list

    def revtok(self, text):
        text = self.re(text)

        return revtok_tokenize(text)

import torch
import torchtext
from torch.autograd import Variable

import re #for build_vocab
from collections import Counter, OrderedDict #for build_vocab

import pprint
pp = pprint.PrettyPrinter(indent=4)

"""a slightly altered version of nestedField
returns a list of tensors rather than stacking them"""

class ListField(torchtext.data.Field):
    """A nested field.
    A nested field holds another field (called *list_field field*), accepts an untokenized
    string or a list string tokens and groups and treats them as one field as described
    by the list_field field. Every token will be preprocessed, padded, etc. in the manner
    specified by the list_field field. Note that this means a nested field always has
    ``sequential=True``. The two fields' vocabularies will be shared. Their
    numericalization results will be stacked into a single tensor. This field is
    primarily used to implement character embeddings. See ``tests/data/test_field.py``
    for examples on how to use this field.
    Arguments:
        list_field (Field): A field contained in this nested field.
        use_vocab (bool): Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: ``True``.
        init_token (str): A token that will be prepended to every example using this
            field, or None for no initial token. Default: ``None``.
        eos_token (str): A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: ``None``.
        fix_length (int): A fixed length that all examples using this field will be
            padded to, or ``None`` for flexible sequence lengths. Default: ``None``.
        tensor_type: The torch.Tensor class that represents a batch of examples
            of this kind of data. Default: ``torch.LongTensor``.
        preprocessing (Pipeline): The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: ``None``.
        postprocessing (Pipeline): A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list,
            the field's Vocab, and train (a bool). Default: ``None``.
        tokenize (callable or str): The function used to tokenize strings using this
            field into sequential examples. If "spacy", the SpaCy English tokenizer is
            used. Default: ``lambda s: s.split()``
        pad_token (str): The string token used as padding. If ``list_field`` is
            sequential, this will be set to its ``pad_token``. Default: ``"<pad>"``.
        pad_first (bool): Do the padding of the sequence at the beginning. Default:
            ``False``.
    """
    def __init__(self, list_field, use_vocab=True, init_token=None, eos_token=None,
                 fix_length=None, tensor_type=torch.LongTensor, preprocessing=None,
                 postprocessing=None, tokenizer=lambda s: s.split(), pad_token='<pad>',
                 pad_first=False, fix_list_length=None):
        if isinstance(list_field, ListField):
            raise ValueError('list_field field must not be another ListField')
        if list_field.include_lengths:
            raise ValueError('list_field field cannot have include_lengths=True')

        if list_field.sequential:
            pad_token = list_field.pad_token
        super(ListField, self).__init__(
            use_vocab=use_vocab,
            init_token=init_token,
            eos_token=eos_token,
            fix_length=fix_length,
            tensor_type=tensor_type,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            lower=list_field.lower,
            tokenize=self.Tokenize_list(tokenizer, self),
            batch_first=True,
            pad_token=pad_token,
            unk_token=list_field.unk_token,
            pad_first=pad_first,
        )
        self.list_field = list_field
        self.vocab_cls = torchtext.vocab.Vocab
        self.fix_list_length = fix_list_length
        self.list_field.fix_length = fix_length

        self.regex_front = re.compile(r'u\"|u\'')
        self.regex_end = re.compile(r'[\'\",]*$')
        self.regex_bracket = re.compile(r'\[|\]')

    def regex(self, x):
        y = self.regex_front.sub("", x)
        y = self.regex_bracket.sub("", y)
        y = self.regex_end.sub("", y)
        return y

    class Tokenize_list(torch.nn.Module):
        def __init__(self, tokenize, outer):
            super(outer.Tokenize_list, self).__init__()
            self.tokenize = tokenize
        def forward(self, input):
            listed = eval(input)
            return [ self.tokenize(ans) for ans in listed ]


    def process(self, batch, device, train):
        """ Process a list of examples to create a torch.Tensor.
        Pad, numericalize, and postprocess a batch and create a tensor.
        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
                and custom postprocessing Pipeline.
        """

        # pad each item
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device, train=train)

        # print(tensor[0].data)
        tensor = [torch.t(lesson) for lesson in tensor]
        arr = torch.stack(tensor, dim=2)  # batch_size, topic_num, topic_words(, char_num)

        # permute for sane reversiblility
        if type(self.list_field) is torchtext.data.NestedField:
            arr = arr.permute(1, 2, 0, 3)  # topic_num, topic_words, batch_size, char_num
        else:
            arr = arr.permute(1, 2, 0)  # topic_num, topic_words, batch_size

        if self.sequential and not self.batch_first:
            arr.t_()
        if device == -1:
            if self.sequential:
                arr = arr.contiguous()
        else:
            arr = arr.cuda(device)

        return arr

    def pad(self, list_batch):
        list_length = max(list_batch, key=lambda x: len(x))
        if self.fix_list_length is not None:
            list_length = self.fix_list_length

        # pad list num
        list_batch = [lesson[:list_length] + [[self.pad_token]] * max(0, list_length - len(lesson)) for lesson in list_batch]

        '''
        max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2

        padded, lengths = [], []
        for lesson in list_batch:
            lp = []
            ll = []
            for x in lesson:
                if self.pad_first:
                    lp.append(
                        [self.pad_token] * max(0, max_len - len(x)) +
                        ([] if self.init_token is None else [self.init_token]) +
                        list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                        ([] if self.eos_token is None else [self.eos_token]))
                else:
                    lp.append(
                        ([] if self.init_token is None else [self.init_token]) +
                        list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                        ([] if self.eos_token is None else [self.eos_token]) +
                        [self.pad_token] * max(0, max_len - len(x)))
                    ll.append(len(lp[-1]) - max(0, max_len - len(x)))
            padded.append(lp)
            lengths.append(ll)
        '''

        padded = []
        lengths = []
        for batch in list_batch:
            if self.include_lengths:
                pa, le = self.list_field.pad(batch)
                padded.append(pa)
                lengths.append(le)
            else:
                padded.append(self.list_field.pad(batch))

        if self.include_lengths:
            return (padded, lengths)
        return padded

    def numericalize(self, arr_list, device=None, train=True):
        numer_each = self.list_field.numericalize

        tensor = [numer_each(x, device=device, train=train) for x in arr_list]

        return torch.stack(tensor, dim=2)

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for listing field and combine it with this field's vocab.
        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for the listing field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        sources = []
        for arg in args:
            if isinstance(arg, torchtext.data.Dataset):
                sources.extend(
                    [getattr(arg, name) for name, field in arg.fields.items()
                     if field is self]
                )
            else:
                sources.append(arg)

        flattened = []
        stripped = []
        for source in sources:
            flattened.extend(source)
        for ex in flattened:
            for ans in ex:
                for word in ans:
                    stripped.append(self.regex(word))

        #pp.pprint(stripped)

        counter = Counter()
        counter.update(stripped)
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

        def extend_vocab(v_parent, v_child, **kwargs):
            v_origin = v_parent
            for word in v_child.itos:
                if word not in v_origin.stoi:
                    v_origin.itos.append(word)
                    v_origin.stoi[word] = len(v_origin.itos) -1
            v_origin.freqs = v_origin.freqs + v_child.freqs

            old_vectors = None
            old_unk_init = None
            old_vectors_cache = None
            if "vectors" in kwargs.keys():
                old_vectors = kwargs["vectors"]
                kwargs["vectors"] = None
            if "unk_init" in kwargs.keys():
                old_unk_init = kwargs["unk_init"]
                kwargs["unk_init"] = None
            if "vectors_cache" in kwargs.keys():
                old_vectors_cache = kwargs["vectors_cache"]
                kwargs["vectors_cache"] = None

            if old_vectors is not None:
                v_origin.load_vectors(old_vectors, unk_init=old_unk_init, cache=old_vectors_cache)

            return v_origin

        self.vocab = extend_vocab(self.vocab, self.list_field.vocab, **kwargs)
        self.list_field.vocab = self.vocab

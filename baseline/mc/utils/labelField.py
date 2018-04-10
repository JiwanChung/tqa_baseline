import torch
import torchtext

"""Forward Compatibility module"""


class LabelField(torchtext.data.Field):
    """A Label field.
    A label field is a shallow wrapper around a standard field designed to hold labels
    for a classification task. Its only use is to set the unk_token and sequential to
    `None` by default.
    """
    def __init__(self, **kwargs):
        # whichever value is set for sequential and unk_token will be overwritten
        kwargs['sequential'] = False
        kwargs['unk_token'] = None

        super(LabelField, self).__init__(**kwargs)
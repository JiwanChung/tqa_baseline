import torchtext

"""Forward Compatibility module"""


class IdField(torchtext.data.Field):
    """A Id field.
    A label field is a shallow wrapper around a standard field designed to hold labels
    for a classification task. Its only use is to set the unk_token and sequential to
    `None` by default.
    """
    def __init__(self, **kwargs):
        # whichever value is set for sequential and unk_token will be overwritten
        kwargs['sequential'] = False
        kwargs['unk_token'] = None

        super(IdField, self).__init__(**kwargs)

    def process(self, batch, device, train):

        return batch

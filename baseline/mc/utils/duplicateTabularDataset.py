from torchtext.data import Dataset
from torchtext.data import Example
from torchtext.utils import unicode_csv_reader

import io
import os
from functools import partial


class DuplicateTabularDataset(Dataset):
    """Defines a Dataset of columns stored in CSV, TSV, or JSON format."""

    def __init__(self, path, format, fields, skip_header=False, **kwargs):
        """Create a DuplicateTabularDataset given a path, file format, and field list.
        A field in the example file can be mapped to mutiple field objects, breaking functional constraint.
        Arguments:
            path (str): Path to the data file.
            format (str): The format of the data file. One of "CSV", "TSV", or
                "JSON" (case-insensitive).
            fields dict[str: tuple(tuple(str, Field), ...)]:
                If using a dict, the keys should be a subset of the JSON keys or CSV/TSV
                columns, and the values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON/CSV/TSV key names
                and also enables selecting a subset of columns to load.
            skip_header (bool): Whether to skip the first line of the input file.
        """
        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromCSV, 'csv': Example.fromCSV}[format.lower()]

        print(path)

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            if format == 'csv':
                reader = unicode_csv_reader(f)
            elif format == 'tsv':
                reader = unicode_csv_reader(f, delimiter='\t')
            else:
                reader = f

            if format in ['csv', 'tsv']:
                if skip_header:
                    raise ValueError('When using a dict to specify fields with a {} file,'
                                     'skip_header must be False and'
                                     'the file must have a header.'.format(format))
                header = next(reader)
                print(header)
                print(fields.keys())
                field_to_index = {f: header.index(f) for f in fields.keys()}
                make_example = partial(make_example, field_to_index=field_to_index)

            if skip_header:
                next(reader)

            examples = [make_example(line, fields) for line in reader]

            super(DuplicateTabularDataset, self).__init__(examples, fields, **kwargs)

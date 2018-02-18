from dpipe.medim.utils import load_by_ids
import pdp


def simple_iterator(ids, load_x, load_y, batch_size, *, shuffle=False):
    def simple():
        for x, y in load_by_ids(load_x, load_y, ids=ids, shuffle=shuffle):
            yield x, y

    return pdp.Pipeline(pdp.Source(simple(), buffer_size=5),
                        pdp.Many2One(chunk_size=batch_size, buffer_size=2),
                        pdp.One2One(pdp.combine_batches, buffer_size=3))
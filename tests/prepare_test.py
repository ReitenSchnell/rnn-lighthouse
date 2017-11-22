from lighthouse.prepare import prepare_data, build_vocabulary, create_batches


def test_data_preparation():
    result = prepare_data('tests/test_data')
    assert result == ['foo', 'boo', 'moo', 'foo', '2222', '1111', 'boo', '2222', 'foo', '3333']


def test_vocabulary_builder():
    result = build_vocabulary(['foo', 'boo', 'moo', 'foo', '2222', '1111', 'boo', '2222', 'foo', '3333'])
    assert result == {'1111': 0, '2222': 1, '3333': 2, 'boo': 3, 'foo': 4, 'moo': 5}


def test_batches():
    text = ['foo', 'boo', 'moo', 'foo', '2222', '1111', 'boo', '2222', 'foo', '3333', 'foo', 'moo', 'boo', '1111', '3333', '2222']
    vocabulary = {'1111': 0, '2222': 1, '3333': 2, 'boo': 3, 'foo': 4, 'moo': 5}
    batch_size = 3
    sec_length = 2
    x, y = create_batches(vocabulary, text, batch_size, sec_length)
    batches_count = int(len(text)/(batch_size * sec_length))
    assert len(x) == len(y) == batches_count
    assert len(x[0]) == len(x[1]) == batch_size
    assert len(y[0]) == len(y[1]) == batch_size
    assert len(x[0][0]) == sec_length
    assert len(y[0][0]) == sec_length





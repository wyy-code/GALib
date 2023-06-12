from abc import ABCMeta

class NetworkAlignmentModel:
    __metaclass__ = ABCMeta

    def __init__(self, source_dataset, target_dataset):
        '''Initialize the Embedding class

        Args:
            source_dataset: source dataset for the alignment
            target_dataset: target dataset for the alignment
        '''
        pass

    def align(self):
        '''Align the source and target dataset, generate an alignment matrix. '''

        pass

    def get_alignment_matrix(self):
        ''' Returns the generated alignment matrix
        Return:
            A numpy array of size #nodes * d
        '''
        pass

    def get_source_embedding(self):
        ''' Returns the learnt embedding of source dataset (if the method generate the embedding)

        Return:
            A numpy array of size #nodes * d
        '''
        return None

    def get_target_embedding(self):
        ''' Returns the learnt embedding of target dataset (if the method generate the embedding)

        Return:
            A numpy array of size #nodes * d
        '''
        return None

	
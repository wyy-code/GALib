from abc import ABCMeta

class NADecoder:
    __metaclass__ = ABCMeta

    def __init__(self, encoder_alignment_matrix):
        '''Initialize the Embedding class

        Args:
            encoder_alignment_matrix: alignment matrix from encoder
        '''
        pass

    def refine_align(self):
        '''Refine the alignment matrix from encoder or NA model, generate an refinement alignment matrix. '''

        pass

    def get_alignment_matrix(self):
        ''' Returns the generated alignment matrix
        Return:
            A numpy array of size #nodes * d
        '''
        pass
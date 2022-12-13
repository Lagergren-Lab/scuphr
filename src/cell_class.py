class Cell(object):
    def __init__(self, r, q, x, y):
        """
        This function generates a Cell object.
        Input: R: array (lc,2) Q: array (lc,2), X: int, Y: int
        """
        self.reads = r  # .astype(int) # Reads
        self.p_error = q    # Error probabilities
        self.Y = int(y)     # Genotype indicator. 0 for no mutation, 1 for mutation
        self.X = int(x)     # Dropout indicator. 0: no ADO. 1: ADO on allele1. 2: ADO on allele2. 3: ADO on both.

        if self.X == 3:
            self.lc = 0  # Read count
        else:
            self.lc = r.shape[0]

    def printCell(self):
        print("\nMutation status of cell: %d" % self.Y)
        print("Allelic dropout status of cell: %d" % self.X)
        print("Total number of reads of the cell: %d" % self.lc)
        print("Reads of cell:")
        print(self.reads)
        print("Error probabilities of cell:")
        print(self.p_error)

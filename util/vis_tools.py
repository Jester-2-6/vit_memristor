import numpy as np

class CSV_Writer:
    def __init__(self, filename: str, header=None):
        self.filename = filename
        self.header = header
        self.file = open(self.filename, 'w')

        if self.header:
            self.file.write(self.header + '\n')

    def write(self, arr):
        if isinstance(arr, np.ndarray):
            if len(arr.shape) > 1:
                for row in arr:
                    self.write(row)
            else:
                arr = ','.join([str(x) for x in arr])
                self.write(arr)
                self.file.write('\n')
        else:
            self.file.write(arr)

    def close(self):
        self.file.close()
        
if __name__ == '__main__':
    x = np.zeros((10,))
    y = x.reshape(x.shape[0], 1)
    
    writer = CSV_Writer('testx.csv')
    writer.write(x)
    writer.close() 
    
    writer = CSV_Writer('testy.csv')
    writer.write(y)
    writer.close() 

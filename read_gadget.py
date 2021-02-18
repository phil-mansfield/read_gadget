import struct

HEADER_SIZE = 0

def LGadget2Cosmological(object):
    def __init__(self, file_name):
        self.f = 

        header_size = struct.unpack(f, "L")
        print(header_size)
        exit(0)
        
        self._n_part = struct.unpack(f, "LLLLLL")
        self._mass = struct.unpack(f, "dddddd")
        self._time = struct.unpack(f, "d")
        self._redshfit = stuct.unpack(f, "d")
    


def test():
    test_file = pass
    
if __name__ == "__main__":
    test()

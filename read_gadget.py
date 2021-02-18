# MIT License

# Copyright (c) 2021 Phil Mansfield

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import division, print_function

import numpy as np
import struct
import array

HEADER_SIZE = 256
RECOGNIZED_VAR_TYPES = ["x", "v", "id", "phi", "acc", "dt"]

class LGadget2(object):
    def __init__(self, file_name):
        """ LGadget2 represents a LGadget2 file located at file_name. It can
        read particle data and contains useful informaiton about the simulation.

        Fields:
              n - The number of particles in the file.
          n_tot - The total number of particles in the simulation.
             mp - The particle mass in Msun/h.
              z - The redshift.
              a - The scale factor.
        omega_m - Omega_M(z=0)
        omega_l - Omega_Lambda(z=0)
           h100 - H(z=0) / (100 km/s/Mpc)
        """

        self.file_name = file_name
        f = open(file_name, "rb")

        # Check that this is the right type of file.
        header_size = struct.unpack("I", f.read(4))[0]
        if header_size != HEADER_SIZE:
            raise ValueError("%s is not an LGadget2 file." % file_name)

        # Read the raw header data.
        self._n_part = struct.unpack("IIIIII", f.read(4*6))
        self._mass = struct.unpack("dddddd", f.read(8*6))
        self._time = struct.unpack("d", f.read(8))[0]
        self._redshift = struct.unpack("d", f.read(8))[0]
        self._flag_sfr = struct.unpack("I", f.read(4))[0]
        self._flag_feedback = struct.unpack("I", f.read(4))[0]
        self._n_part_total = struct.unpack("IIIIII", f.read(4*6))
        self._box_size = struct.unpack("d", f.read(8))[0]
        self._omega0 = struct.unpack("d", f.read(8))[0]
        self._omega_lambda = struct.unpack("d", f.read(8))[0]
        self._hubble_param = struct.unpack("d", f.read(8))[0]
        self._flag_setllar_age = struct.unpack("I", f.read(4))[0]
        self._hash_tab_size = struct.unpack("I", f.read(4))[0]

        # Convert to, IMO, more user-friendly versions of the default header
        # fields.
        self.n = self._n_part[1] + (self._n_part[0]<<32)
        self.n_tot = self._n_part_total[1] + (self._n_part_total[0]<<32)
        self.mp = self._mass[1]*1e10
        self.z = self._redshift
        self.a = self._time
        self.omega_m = self._omega0
        self.omega_l = self._omega_lambda
        self.h100 = self._hubble_param

        f.close()

    def read(self, var_type, fields=("x", "v", "id")):
        """ read() reads the specified variable from the file. For a standard
        file, the fields are:
          "x" - Position vectors in comoving Mpc/h. Shape: (n, 3)
          "v" - Velocity vectors in physical km/s. Shape: (n, 3)
         "id" - Unique integer ID identifying eahc particle. Shape: (n, 3)

        Some files can also contian additional information:
        "phi" - Potential. No clue what the units are. Shape: (n,)
        "acc" - Acceleration. No clue what the units are. Shape: (n, 3)
         "dt" - Local timestep size in delta ln(a(z)). Shape: (n,)

        If you are using these additional fields, you can specify their order
        in the file by setting the fields argument to a tuple of strings in the
        correct order.
        """

        if var_type not in RECOGNIZED_VAR_TYPES:
            raise ValueError("Cannot read var_type %s, only %s are valid" %
                             (var_type, RECOGNIZED_VAR_TYPES))
        if var_type not in fields:
            raise ValueError("var_type %s not in file's fields %s" %
                             (var_type, fields))

        # Compute offset to data.
        offset = HEADER_SIZE + 12
        for i in range(len(fields)):
            if fields[i] == var_type: break
            offset += self._field_size(var_type)
            
        f = open(self.file_name, "rb")
        f.seek(offset)

        # Read and convert out of code units.
        if var_type == "x":
            x = array.array("f")
            x.fromfile(f, self.n*3)
            x = np.array(x, dtype=np.float32)
            x = x.reshape((self.n, 3))
            return x

        elif var_type == "v":
            v = array.array("f")
            v.fromfile(f, self.n*3)
            v = np.array(v, dtype=np.float32)
            v = v.reshape((self.n, 3))
            v *= np.sqrt(self.a)
            return v

        elif var_type == "id":
            id = array.array("Q")
            id.fromfile(f, self.n)
            return np.array(id, dtype=np.uint64)

        elif var_type == "phi":
            phi = array.array("f")
            phi.fromfile(f, self.n)
            return np.array(phi, dtype=np.float32)

        elif var_type == "acc":
            acc = array.array("f")
            acc.fromfile(f, self.n*3)
            acc = np.array(acc, dtype=np.float32)
            acc = acc.reshape((self.n, 3))
            return acc

        elif var_type == "dt":
            dt = array.array("f")
            dt.fromfile(f, self.n)
            return np.array(dt, dtype=np.float32)

        else:
            assert(0)
        
        f.close()

    def _field_size(self, var_type):
        elem_size = {"x": 12, "v": 12, "acc": 12,
                     "phi": 4, "dt": 4, "id": 8}[var_type]
        return 8 + elem_size*self.n

def test():    
    test_file = "/data/mansfield/simulations/Erebos_CBol_L63/particles/raw/snapdir_100/snapshot_100.0"

    f = LGadget2(test_file)

    x = f.read("x")
    v = f.read("v")
    id = f.read("id")

    print(x)
    print(v)
    print(id)
    
if __name__ == "__main__":
    test()

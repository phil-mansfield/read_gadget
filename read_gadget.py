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
    def __init__(self, file_name, fields=("x", "v", "id")):
        """ LGadget2 represents a LGadget2 file located at file_name. It can
        read particle data and contains useful informaiton about the simulation.

        Fields:
              n - The number of particles in the file.
          n_tot - The total number of particles in the simulation.
             mp - The particle mass in Msun/h.
              L - The box size in comoving Mpc/h.
              z - The redshift.
              a - The scale factor.
        omega_m - Omega_M(z=0)
        omega_l - Omega_Lambda(z=0)
           h100 - H(z=0) / (100 km/s/Mpc)

        If you are using non-standard fields, you can specify their order
        in the file by setting the fields argument to a tuple of strings in the
        correct order. See the read docstring for addtional details.
        """

        self.fields = fields
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
        self.L = self._box_size
        self.mp = self._mass[1]*1e10
        self.z = self._redshift
        self.a = self._time
        self.omega_m = self._omega0
        self.omega_l = self._omega_lambda
        self.h100 = self._hubble_param

        f.close()

    def read(self, var_type):
        """ read() reads the specified variable from the file. For a standard
        file, the fields are:
          "x" - Position vectors in comoving Mpc/h. Shape: (n, 3)
          "v" - Velocity vectors in physical km/s. Shape: (n, 3)
         "id" - Unique integer ID identifying each particle. Shape: (n,)

        Some files can also contian additional information:
        "phi" - Potential. No clue what the units are. Shape: (n,)
        "acc" - Acceleration. No clue what the units are. Shape: (n, 3)
         "dt" - Local timestep size in delta ln(a(z)). Shape: (n,)
        """

        if var_type not in RECOGNIZED_VAR_TYPES:
            raise ValueError("Cannot read var_type %s, only %s are valid" %
                             (var_type, RECOGNIZED_VAR_TYPES))
        if var_type not in self.fields:
            raise ValueError("var_type %s not in file's fields %s" %
                             (var_type, self.fields))

        # Compute offset to data.
        offset = HEADER_SIZE + 12
        for i in range(len(fields)):
            if self.fields[i] == var_type: break
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

class Gadget2Zoom(object):
    def __init__(self, file_name, fields=("x", "v", "id")):
        """ Gadget2Zoom represents a Gadget2 file located at file_name which is
        associated with a multi-reoslution zoom-in simulation. It can read
        particle data and contains useful informaiton about the simulation.

        Fields:
              n - The number of particles in the file. An array where each entry
                  is a different resolution level. empty resolution levels
                  have been removed and the array is ordered from smallest to
                  largest particle size.  
          n_tot - The total number of particles in the simulation. An array with
                  the same shape as n.
         n_file - The number of files in the simulation.
             mp - The particle mass in Msun/h. An array with the same shape as
                  n.
              L - The box size in comoving Mpc/h.
              z - The redshift.
              a - The scale factor.
        omega_m - Omega_M(z=0)
        omega_l - Omega_Lambda(z=0)
           h100 - H(z=0) / (100 km/s/Mpc)

        If you are using non-standard fields, you can specify their order
        in the file by setting the fields argument to a tuple of strings in the
        correct order. See the read docstring for addtional details.
        """

        self.fields = fields
        self.file_name = file_name
        f = open(file_name, "rb")

        # Check that this is the right type of file.
        header_size = struct.unpack("I", f.read(4))[0]

        if header_size != HEADER_SIZE:
            raise ValueError("%s is not an Gadget2 file." % file_name)

        # Read the raw header data.
        self._n_part = struct.unpack("IIIIII", f.read(4*6))
        self._mass = struct.unpack("dddddd", f.read(8*6))
        self._time = struct.unpack("d", f.read(8))[0]
        self._redshift = struct.unpack("d", f.read(8))[0]
        self._flag_sfr = struct.unpack("I", f.read(4))[0]
        self._flag_feedback = struct.unpack("I", f.read(4))[0]
        self._n_part_total = struct.unpack("IIIIII", f.read(4*6))
        self._flag_feedback = struct.unpack("I", f.read(4))[0]
        self._num_files = struct.unpack("I", f.read(4))[0]
        self._box_size = struct.unpack("d", f.read(8))[0]
        self._omega0 = struct.unpack("d", f.read(8))[0]
        self._omega_lambda = struct.unpack("d", f.read(8))[0]
        self._hubble_param = struct.unpack("d", f.read(8))[0]
        self._flag_stellar_age = struct.unpack("I", f.read(4))[0]
        self._flag_metals = struct.unpack("I", f.read(4))[0]
        self._n_part_total_hw = struct.unpack("IIIIII", f.read(4*6))
        self._flag_entropy_ics = struct.unpack("I", f.read(4))

        # Convert to, IMO, more user-friendly versions of the default header
        # fields.
        self.n = np.array(self._n_part, dtype=np.int64)
        self.n_tot = np.array(self._n_part_total, dtype=np.int64)
        for i in range(len(self.n_tot)):
            self.n_tot[i] += self._n_part_total_hw[i] << 32
        self.mp = np.array(self._mass)*1e10
            
        ok = self.n_tot > 0
        self.n, self.n_tot, self.mp = self.n[ok], self.n_tot[ok], self.mp[ok]

        self.L = self._box_size
        self.z = self._redshift
        self.a = self._time
        self.omega_m = self._omega0
        self.omega_l = self._omega_lambda
        self.h100 = self._hubble_param
        self.n_files = self._num_files

        f.close()

    def read(self, var_type):
        """ read() reads the specified variable from the file. For a standard
        file, the fields are:
          "x" - Position vectors in comoving Mpc/h. Shape: (n, 3)
          "v" - Velocity vectors in physical km/s. Shape: (n, 3)
         "id" - Unique integer ID identifying each particle. Shape: (n,)

        Some files can also contian additional information:
        "phi" - Potential. No clue what the units are. Shape: (n,)
        "acc" - Acceleration. No clue what the units are. Shape: (n, 3)
         "dt" - Local timestep size in delta ln(a(z)). Shape: (n,)
        """

        if var_type not in RECOGNIZED_VAR_TYPES:
            raise ValueError("Cannot read var_type %s, only %s are valid" %
                             (var_type, RECOGNIZED_VAR_TYPES))
        if var_type not in self.fields:
            raise ValueError("var_type %s not in file's fields %s" %
                             (var_type, self.fields))

        # Compute offset to data.
        offset = HEADER_SIZE + 12
        for i in range(len(self.fields)):
            if self.fields[i] == var_type: break
            offset += self._field_size(var_type)
            
        f = open(self.file_name, "rb")
        f.seek(offset)

        n = np.sum(self.n)

        # Read and convert out of code units.
        if var_type == "x":
            x = array.array("f")
            x.fromfile(f, n*3)
            x = np.array(x, dtype=np.float32)
            x = x.reshape((n, 3))
            return self._resolution_bins(x)

        elif var_type == "v":
            v = array.array("f")
            v.fromfile(f, n*3)
            v = np.array(v, dtype=np.float32)
            v = v.reshape((n, 3))
            v *= np.sqrt(self.a)
            return self._resolution_bins(v)

        elif var_type == "id":
            id = array.array("Q")
            id.fromfile(f, n)
            return self._resolution_bins(np.array(id, dtype=np.uint64))

        elif var_type == "phi":
            phi = array.array("f")
            phi.fromfile(f, n)
            return self._resolution_bins(np.array(phi, dtype=np.float32))

        elif var_type == "acc":
            acc = array.array("f")
            acc.fromfile(f, n*3)
            acc = np.array(acc, dtype=np.float32)
            acc = acc.reshape((n, 3))
            return self._resolution_bins(acc)

        elif var_type == "dt":
            dt = array.array("f")
            dt.fromfile(f, n)
            return self._resolution_bins(np.array(dt, dtype=np.float32))

        else:
            assert(0)
        
        f.close()

    def _field_size(self, var_type):
        elem_size = {"x": 12, "v": 12, "acc": 12,
                     "phi": 4, "dt": 4, "id": 8}[var_type]
        return 8 + elem_size*np.sum(self.n)

    def _resolution_bins(self, x):
        end = np.cumsum(self.n)
        start = np.zeros(len(end), dtype=int)
        start[1:] = end[:-1]

        out = [None]*len(end)
        for i in range(len(end)):
            out[i] = x[start[i]: end[i]]

        return out

def test():    
    test_file = "/scratch/users/enadler/new_mw_zoomins/Halo416_2K/output_potential/snapshot_000.0"

    f = Gadget2Zoom(test_file, ("x", "v", "id", "phi"))

    x = f.read("x")
    v = f.read("v")
    id = f.read("id")
    phi = f.read("phi")

    print("High resolution x")
    print(x[0])
    print("High resolution v")
    print(v[0])
    print("High resolution id")
    np.set_printoptions(formatter={'int':hex})
    print(id[0])
    print("High resolution phi")
    print(phi[0])
    
if __name__ == "__main__":
    test()

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
RECOGNIZED_VAR_TYPES = ["x", "v", "id32", "id64", "phi", "acc", "dt"]

class AbstractCosmological(object):
    def read(self, var_type):
        """ read() reads the specified variable from the file. For a standard
        file, the fields are:
          "x" - Position vectors in comoving Mpc/h. Shape: (n, 3)
          "v" - Velocity vectors in physical km/s. Shape: (n, 3)
         "id64" - Unique uint64 ID identifying each particle. Shape: (n,)
         "id32" - Unique uint32 ID identifying each particle. Shape: (n,)

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
            offset += self._field_size(self.fields[i])
            
        f = open(self.file_name, "rb")
        f.seek(offset - 4)

        size = struct.unpack("I", f.read(4))[0]
        expected_size = {
            "x": 12, "v": 12, "id64": 8, "id32": 4,
            "phi": 4, "acc": 12, "dt": 4
        }[var_type]*self.n
        if size != expected_size:
            raise ValueError(
                "%s block should have size %d, but actual size is %d" % 
                (var_type, expected_size, size)
            )

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

        elif var_type == "id64":
            id = array.array("Q")
            id.fromfile(f, self.n)
            return np.array(id, dtype=np.uint64)

        elif var_type == "id32":
            id = array.array("I")
            id.fromfile(f, self.n)
            return np.array(id, dtype=np.uint32)

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
        elem_size = {"x": 12, "v": 12, "acc": 12, "id32": 4,
                     "phi": 4, "dt": 4, "id64": 8}[var_type]
        return 8 + elem_size*self.n

class LGadget2(AbstractCosmological):
    # See AbstractCosmological for the read() method documentaion.
    def __init__(self, file_name, fields=("x", "v", "id64")):
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
        self._flag_cooling = struct.unpack("I", f.read(4))[0]
        self._num_files = struct.unpack("I", f.read(4))[0]
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

class Gadget2Cosmological(AbstractCosmological):
    # See AbstractCosmological for the read() method documentaion.
    def __init__(self, file_name, fields=("x", "v", "id64")):
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

        self.n = self._n_part[1]
        self.n_tot = (int(self._n_part_total[1]) +
                      (int(self._n_part_total_hw[1])<<32))

        print(self._n_part)
        print(self._n_part_total)
        print(self._n_part_total_hw)

        self.mp = np.array(self._mass[1])*1e10
        
        self.L = self._box_size
        self.z = self._redshift
        self.a = self._time
        self.omega_m = self._omega0
        self.omega_l = self._omega_lambda
        self.h100 = self._hubble_param
        self.n_files = self._num_files

        f.close()

        
class Gadget2Zoom(object):
    def __init__(self, file_name, fields=("x", "v", "id64")):
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
        self.n = np.array(self._n_part, dtype=np.uint64)
        self.n_tot = np.array(self._n_part_total, dtype=np.uint64)
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
        "id64" - Unique uint64 ID identifying each particle. Shape: (n,)
        "id32" - Unique uint32 ID identifying each particle. Shape: (n,)

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
            offset += self._field_size(self.fields[i])
            
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

        elif var_type == "id64":
            id = array.array("Q")
            id.fromfile(f, n)
            return self._resolution_bins(np.array(id, dtype=np.uint64))

        elif var_type == "id32":
            id = array.array("I")
            id.fromfile(f, n)
            return self._resolution_bins(np.array(id, dtype=np.uint32))

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
        elem_size = {"x": 12, "v": 12, "acc": 12, "id32": 4,
                     "phi": 4, "dt": 4, "id64": 8}[var_type]
        return 8 + elem_size*np.sum(self.n)

    def _resolution_bins(self, x):
        end = np.cumsum(self.n)
        start = np.zeros(len(end), dtype=int)
        start[1:] = end[:-1]

        out = [None]*len(end)
        for i in range(len(end)):
            out[i] = x[start[i]: end[i]]

        return out

def music_index_from_file(file_name):
    """ music_index_from_file returns a MusicIndex object based off the parsed
    output log (a conf_log file) from MUSIC. Most users will want to use this 
    instead of creating MusicIndex objects by hand.
    """
    with open(file_name) as fp: text = fp.read()
    lines = text.split("\n")

    if not _is_music_log(lines): _parse_fail(file_name)

    domain_dx, ok = _parse_vector(lines[13])
    if not ok: _parse_fail(file_name)
    
    level, offset, size = [], [], []
    for i in range(14, len(lines), 2):
        level_i, ok = _parse_level(lines[i])
        if not ok: break
        offset_i, ok = _parse_vector(lines[i])
        if not ok: _parse_fail(file_name)
        size_i, ok = _parse_vector(lines[i+1])

        level.append(level_i)
        offset.append(offset_i)
        size.append(size_i)

    level = np.array(level)[::-1]
    offset = np.array(offset)[::-1,:]
    size = np.array(size)[::-1,:]
            
    return MusicIndex(domain_dx, level, offset, size)

def _is_music_log(lines):
    return len(lines) >= 16 and "Domain shifted by" in lines[13]

def _parse_fail(file_name):
    raise ValueError("%s is not a valid music conf_log file." % file_name)

def _parse_level(line):
    tokens = [tok for tok in line.split(" ") if len(tok) > 0]

    try:
        i = tokens.index(":")
        return int(tokens[i - 1]), True
    except:
        return None, False

def _parse_vector(line):
    tokens = [tok for tok in line.split(" ") if len(tok) > 0]
    try:
        i = tokens.index("(")
    except:
        return None, False
    if i + 3 >= len(tokens): return None, False
    
    vec = np.zeros(3, dtype=int)
    for k in range(3):
        try:
            tok = tokens[i + 1 + k]
            if tok[-1] != "," and tok[-1] != ")": return None, False
            vec[k] = int(tok[:-1])
        except:
            return None, False

    return vec, True
    
class MusicIndex(object):
    def __init__(self, domain_dx, level, offset, size):
        """ MusicIndex is  an index for looking up the grid coordinates of
        particles generated by teh IC generation code, MUSIC. Most users will
        want to use music_index_from_file() instead of calling this constructor
        directly.

        domain_dx - The top-level shift in lowest-reoslution grid.
            level - The "level" of each grid from highest to lowest resolution. 
                    A level i grid would have 2^i particles on a side if
                    simulated as a full box.
           offset - The offset of the origin of the level i grid from the origin
                    of the level i+1 grid in level i+1 grid units.
             size - The dimensions of the levle i grid in level i grid units.

        The most important method of this object is fine_idx_array.
        """
        self.domain_dx = domain_dx
        self.level = level
        self.offset = offset
        self.size = size

        self.n_logical = size[:,0]*size[:,1]*size[:,2]
        self.n_physical = np.copy(self.n_logical)
        self.n_physical[1:] -= self.n_physical[:-1] // 8

        self.fine_domain_dx = self.domain_dx * 2**(level[0] - level[-1])
        self.fine_offset = np.zeros(self.offset.shape, dtype=int)
        for i in range(len(self.offset) - 2, -1, -1):
            scale = 2**(self.level[0] - self.level[i] + 1)
            self.fine_offset[i,:] = (self.fine_offset[i+1,:] +
                                     scale*self.offset[i,:])
            
    def idx_to_vec(self, L, level, idx_vector, remainder=None):
        """ idx_to_vec converts the grid coordinates a vector of indices
        at a given level from grid units to physical units in box with side
        length L. You amy supply an optional fracitonal remainder for points
        not exactly aligned with the grid.
        """
        dx = L / 2**level
        if remainder is None:
            return dx*idx_vector
        else:
            return dx*(idx_vector + remainder)
        
    def coarse_to_fine(self, level_coarse, level_fine, idx_vector):
        """ coarse_to_fine converts indices from a coarser grid level to a
        finer one.
        """
        return idx_vector * 2**(level_fine - level_coarse)
        
    def fine_to_coarse(self, level_fine, level_coarse, idx_vector):
        """ fine_to_coarse converts indices from a finer grid level to a
        coarser one.
        """
        return idx_vector // 2**(level_fine - level_coarse)

    def fine_to_coarse_remainder(self, level_fine, level_coarse, idx_vector):
        """ Returns the fracitonal remainder of the conversion from a fine to 
        coarse grid in the grid units of the coarser grid.
        """
        rem = idx_vector % 2**(level_fine - level_coarse)
        return rem / float(2**(level_fine - level_coarse))

    def fine_idx_array(self, level_idxs=None, original_coordinates=False):
        """ fine_idx_array returns the grid indices of particles in the order
        which particles are output by MUSIC (i.e. an ID-based lookup table).
        The indices are in the highest resolution grid units.

        Since this lookup table can be quite memory-intesnive, you can specify
        which levels get included through level_idx. You can also use the
        original_coordinates flag to return indexes in the coordinate system
        of  the original (unshifted) ICs.
        """
        arrays = []
        if level_idxs is None: level_idxs = range(len(self.level))
        for level_i in level_idxs:
            arrays.append(self._fine_idx_array_level(
                level_i, original_coordinates))
            
        return np.hstack(arrays)
        
    def _fine_idx_array_level(self, level_i, original_coordinates):
        idx = np.arange(self.n_logical[level_i], dtype=int)
        # IDs are in z-y-x-major ordering:
        idx_vector = np.array([
            idx // (self.size[level_i,2]*self.size[level_i,1]),
            (idx // self.size[level_i,2]) % self.size[level_i,1],
            idx % self.size[level_i,2],
        ]).T

        if level_i > 0:
            ok = np.zeros(len(idx), dtype=bool)
            for k in range(3):
                low = self.offset[level_i - 1,k]
                high = self.size[level_i - 1,k] // 2 + low
                ok = ok | (idx_vector[:,k] < low) | (idx_vector[:,k] >= high)

            idx_vector = idx_vector[ok,:]

        assert(idx_vector.shape[0] == self.n_physical[level_i])
            
        for k in range(3):
            idx_vector[:,k] += self.fine_offset[level_i,k]

            if original_coordinates:
                idx_vector[:,k] -= self.fine_domain_dx[k]
            
                too_large = idx_vector[:,k] >= 2**self.level[0]
                too_small = idx_vector[:,k] < 0
                idx_vector[too_large,k] -= 2**self.level[0]
                idx_vector[too_small,k] += 2**self.level[0]
            
        return idx_vector

def test2():
    music_idx = music_index_from_file("example_music_conf.txt")
    idx = music_idx.fine_idx_array(level_idxs=[0], original_coordinates=True)
    vec = music_idx.idx_to_vec(125, music_idx.level[0], idx)
    print(vec)

def test():    
    file_fmt = "/scratch/users/enadler/new_mw_zoomins/Halo416_2K/output_potential/snapshot_000.%d"

    f = Gadget2Zoom(file_fmt % 0, ("x", "v", "id32", "phi", "acc"))

    x = f.read("x")
    v = f.read("v")
    id = f.read("id32")
    phi = f.read("phi")
    acc = f.read("acc")

    hist_range = (-100000, 100000)
    bins = 200
    N = np.zeros((bins, len(phi)))

    mins = [np.min(id_j) for id_j in id]
    maxes = [np.max(id_j) for id_j in id]

    for i in range(8):
        file_name = file_fmt % i
        f = Gadget2Zoom(file_name, ("x", "v", "id32", "phi"))

        phi = f.read("phi")
        id = f.read("id32")

        for j in range(len(phi)):
            maxes[j] = max(np.max(id[j]), maxes[j])
            mins[j] = min(np.min(id[j]), mins[j])
            n, edges = np.histogram(phi[j], range=hist_range, bins=bins)
            N[:,j] += n

    mid = (edges[1:] + edges[:-1]) / 2

    print(f.mp)
    print(f.n_tot)
    print(np.cumsum(f.n_tot))
    print(mins)
    print(maxes)

    """
    for i in range(len(mid)):
        print("%11.3f" % mid[i], end=" ")
        for j in range(len(phi)):
            print("%8d" % N[i,j], end=" ")
        print()
    """
    
if __name__ == "__main__":
    #test()
    test2()

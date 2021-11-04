package read_gadget

import (
	"encoding/binary"
	"fmt"
	"os"
)

type Gadget2ZoomFile struct {
	N, NTot [6]int
	Mp [6]float64
	L, Z, Scale, OmegaM, OmegaL, H100 float64
	Fields []string
	FileName string
}

type gadget2ZoomHeader struct {
	NPart [6]int32
	Mp [6]float64
	Time, Redshfit float64
	FlagSFR, FlagFeedback int32
	NPartTotal [6]int32
	FlagCooling, NumFiles int32
	L, Omega0, OmegaLambda, HubbleParam float64
}

func OpenGadget2Zoom(name string, fields []string) *Gadget2ZoomFile {	
	f, err := os.Open(name)
	if err != nil { panic(err.Error()) }
	defer f.Close()

	_, err = f.Seek(4, 0)
	if err != nil { panic(err.Error()) }

	hd := &gadget2ZoomHeader{ }
	err = binary.Read(f, binary.LittleEndian, hd)
	if err != nil { panic(err.Error()) }

	n, nTot := [6]int{ }, [6]int{ }
	for i := 0; i < 6; i++ {
		n[i], nTot[i] = int(hd.NPart[i]), int(hd.NPartTotal[i])
		hd.Mp[i] *= 1e10
	}
	
	return &Gadget2ZoomFile{
		N: n, NTot: nTot, Mp: hd.Mp,
		L: hd.L, Z: 1/hd.Time - 1, Scale: hd.Time,
		OmegaM: hd.Omega0, OmegaL: hd.OmegaLambda, H100: hd.HubbleParam,
		Fields: fields, FileName: name,
	}
}

func (f *Gadget2ZoomFile) Read(
	varType string, level int, buf ...interface{},
) interface{} {
	off := offset(f.Fields, f.N, varType, level)

	var out interface{}
	if len(buf) > 0 {
		out = buf[0]
	} else {
		switch varType {
		case "id64": out = make([]int64, f.N[level])
		case "id32": out = make([]int32, f.N[level])
		case "x", "v": out = make([][3]float32, f.N[level])
		default: out = make([]float32, f.N[level])
		}
	}

	file, err := os.Open(f.FileName)
	if err != nil { panic(err.Error()) }

	_, err = file.Seek(off, 0)
	if err != nil { panic(err.Error()) }
	
	err = binary.Read(file, binary.LittleEndian, out)
	if err != nil { panic(err.Error()) }
	
	return out
}

func offset(fields []string, n [6]int, varType string, level int) int64 {
	off := int64(8 + 256)

	nTot := 0
	for i := range n { nTot += n[i] }

	for i := range fields {
		if fields[i] == varType {
			off += 4
			for j := 0; j < level; j++ { 
				off += int64(n[j])*varSize(fields[i])
			}
			fmt.Println(varType, level, off)
			return off
		} else {
			off += varSize(fields[i])*int64(nTot) + 8
		}
	}

	panic(fmt.Sprintf("var %s not in fields %s", varType, fields))
}

func varSize(varType string) int64 {
	switch varType {
	case "id64": return 8
	case "x", "v", "acc": return 12
	default: return 4
	}
}

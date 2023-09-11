import flatbuffers

# automatically generated by the FlatBuffers compiler, do not modify

# namespace: preprocessor

from flatbuffers.compat import import_numpy
np = import_numpy()

class Data(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Data()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsData(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Data
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Data
    def InputModelPath(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def DataStart(builder): builder.StartObject(1)
def Start(builder):
    return DataStart(builder)
def DataAddInputModelPath(builder, inputModelPath): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(inputModelPath), 0)
def AddInputModelPath(builder, inputModelPath):
    return DataAddInputModelPath(builder, inputModelPath)
def DataEnd(builder): return builder.EndObject()
def End(builder):
    return DataEnd(builder)

class DataT(object):

    # DataT
    def __init__(self):
        self.inputModelPath = None  # type: str

    @classmethod
    def InitFromBuf(cls, buf, pos):
        data = Data()
        data.Init(buf, pos)
        return cls.InitFromObj(data)

    @classmethod
    def InitFromObj(cls, data):
        x = DataT()
        x._UnPack(data)
        return x

    # DataT
    def _UnPack(self, data):
        if data is None:
            return
        self.inputModelPath = data.InputModelPath()

    # DataT
    def Pack(self, builder):
        if self.inputModelPath is not None:
            inputModelPath = builder.CreateString(self.inputModelPath)
        DataStart(builder)
        if self.inputModelPath is not None:
            DataAddInputModelPath(builder, inputModelPath)
        data = DataEnd(builder)
        return data

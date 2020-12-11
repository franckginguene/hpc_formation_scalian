include(../config.qmake)

TEMPLATE = app

SOURCES += *.cu  // Dirty trick to add the CU files in the QTCreator source files.
SOURCES -= *.cu  // Dirty trick to add the CU files in the QTCreator source files.

CUDA_SOURCES += *.cu

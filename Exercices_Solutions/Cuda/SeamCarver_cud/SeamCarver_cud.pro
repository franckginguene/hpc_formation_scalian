include(../config.qmake)

TEMPLATE = app

SOURCES += *.cu  // Dirty trick to add the CU files in the QTCreator source files.
SOURCES -= *.cu  // Dirty trick to add the CU files in the QTCreator source files.

HEADERS += *.h

CUDA_SOURCES += *.cu

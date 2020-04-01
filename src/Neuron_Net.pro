TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
LIBS += -lGLEW -lglfw3 -lGL -lX11 -lXi -lXrandr -lXxf86vm -lXinerama -lXcursor -lrt -lm -pthread -ldl -lSOIL


SOURCES += \
    main.cpp \
    dataframe.cpp \
    neuro_net.cpp

HEADERS += \
    dataframe.h \
    neuro_net.h

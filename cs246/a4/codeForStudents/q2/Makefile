CXX = clang++
CXXFLAGS = -std=c++20 -Wall -MMD
EXEC = a4q2
SOURCES = $(wildcard *.cc)
OBJECTS = $(patsubst %.cc, %.o, $(SOURCES))
DEPENDS = ${OBJECTS:.o=.d}

${EXEC}: ${OBJECTS}
	${CXX} ${CXXFLAGS} ${OBJECTS} -o ${EXEC}

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

-include ${DEPENDS}

.PHONY: clean

clean:
	rm ${OBJECTS} ${EXEC} ${DEPENDS}

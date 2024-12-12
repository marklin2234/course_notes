CXX = g++-11
CXXFLAGS = -std=c++20 -Wall -MMD -g
EXEC = exc
OBJECTS = main.o observer.o subject.o assignment.o cs246.o student.o
DEPENDS = ${OBJECTS:.o=.d}

${EXEC}: ${OBJECTS}
	${CXX} ${CXXFLAGS} ${OBJECTS} -o ${EXEC}

-include ${DEPENDS}

.PHONY: clean

clean:
	rm ${OBJECTS} ${DEPENDS}

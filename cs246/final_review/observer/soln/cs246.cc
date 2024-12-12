#include "cs246.h"

CS246::CS246(): currentAsst{} {}

CS246::CS246(std::string name, float weight): currentAsst{name, weight} {}

void CS246::newAssignment(string name, float weight) {
    currentAsst.updateAssignment(name, weight);
    notifyObservers();
}

Assignment CS246::getAssignment() { return currentAsst; }

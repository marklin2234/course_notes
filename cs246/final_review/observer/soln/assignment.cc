#include "assignment.h"

Assignment::Assignment(): name{""}, weight{0} {}

Assignment::Assignment(string name, float weight): name{name}, weight{weight} {}

void Assignment::updateAssignment(string newName, float newWeight) {
    name = newName;
    weight = newWeight;
}

string Assignment::getName() const { return name; }

float Assignment::getWeight() const { return weight; }


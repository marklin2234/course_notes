#ifndef _ASSIGNMENT_H_
#define _ASSIGNMENT_H_

#include <string>

using namespace std;

class Assignment {  // only a data holder
    string name;
    float weight;  // 0 <= weight <= 1
  public:
    Assignment();
    Assignment(string name, float weight);
    void updateAssignment(string newName, float newWeight);
    string getName() const;
    float getWeight() const;
};

#endif

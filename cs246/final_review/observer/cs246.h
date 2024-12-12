#ifndef _CS246_H_
#define _CS246_H_

#include <string>
#include "subject.h"
#include "assignment.h"

class CS246 : public Subject {
    Assignment currentAsst;
  public:
    CS246();
    CS246(std::string name, float weight);
    void newAssignment(string name, float weight);
    Assignment getAssignment();
};

#endif

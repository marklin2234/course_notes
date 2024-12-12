#include "student.h"

using namespace std;

Student::Student(shared_ptr<CS246>course, string name, float knowledge):
  course{course}, name{name}, knowledge{knowledge}, grade{0} {
    course->add(this);
    cout << name << " is now enrolled!" << endl;
}

Student::~Student() { course->drop(this); }

void Student::study() { knowledge = ((knowledge + 0.1) >= 1)? 1 : (knowledge + 0.1); }

void Student::doHomework() {
    Assignment homework = course->getAssignment();
    grade += (homework.getWeight() * knowledge * 100);
    cout << name << " completed " << homework.getName() << "! Total grade: " << grade << endl;
}

string Student::getName() const { return name; }

float Student::getKnowledge() const { return knowledge; }

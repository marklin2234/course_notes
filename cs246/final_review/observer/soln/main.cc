#include <iostream>
#include <vector>
#include <memory>
#include "observer.h"
#include "student.h"
#include "cs246.h"

using namespace std;

int main() {
    shared_ptr<CS246> cs246 = make_shared<CS246>();
    vector<Student *> students;
    string cmd, name;
    float knowledge, weight;
    while (cin >> cmd) {
        if (cmd == "s") {  // enroll a new student
            cin >> name;
            cin >> knowledge;
            students.emplace_back(new Student(cs246, name, knowledge));
        } else if (cmd == "a") {  // post a new assignment
            cin >> name;
            cin >> weight;
            cs246->newAssignment(name, weight);
        } else if (cmd == "study") {  // let a student study
            cin >> name;
            for (auto s : students) {
                if (s->getName() == name) {
                    s->study();
                    cout << s->getName() << " is studying hard! Current C++ knowledge: " << s->getKnowledge() << endl;
                    break;
                }
            }
        } else if (cmd == "drop") {  // let a student drop the course
            cin >> name;
            unsigned int counter = 0;
            for (auto it = students.begin(); it != students.end(); ++it) {
                if (students[counter]->getName() == name) {
                    cout << name << " has dropped the course." << endl;
                    delete students[counter];
                    students.erase(it);
                    break;
                }
                ++counter;
            } 
        } else if (cmd == "q") {  // quit
            break;
        } else {
            cerr << "Invalid command: " << cmd << endl;
            continue;
        }
    }
    for (auto s : students) delete s;
}

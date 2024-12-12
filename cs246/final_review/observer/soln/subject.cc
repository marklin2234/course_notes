#include "subject.h"

Subject::Subject() {}
Subject::~Subject() {}

void Subject::add(Observer *o) { observers.emplace_back(o); }

void Subject::drop(Observer *o) {  // assume the student is unique
    for (auto it = observers.begin(); it != observers.end(); ++it) {
        if (*it == o) {
            observers.erase(it); 
            break;
        }
    }
}

void Subject::notifyObservers() const {
    for (auto ob : observers) ob->doHomework();
}


#include "subject.h"

Subject::Subject() {}
Subject::~Subject() {}

void Subject::add(Observer *o) {  // i.e. the attach() method
    observers.emplace_back(o);
}

void Subject::drop(Observer *o) {
    for (auto it = observers.begin(); it != observers.end(); it++) {
        observers.erase(it);
    }
}

void Subject::notifyObservers() const {
    for (const auto &o : observers) {
        o->doHomework();
    }
}


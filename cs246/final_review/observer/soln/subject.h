#ifndef _SUBJECT_H_
#define _SUBJECT_H_

#include <vector>
#include "observer.h"

class Subject {
  std::vector<Observer*> observers;

 public:
  Subject();
  void add(Observer *o);
  void drop(Observer *o);
  void notifyObservers() const;
  virtual ~Subject() = 0;
};


#endif

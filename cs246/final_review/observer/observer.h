#ifndef _OBSERVER_H_
#define _OBSERVER_H_

class Observer {
 public:
  virtual void doHomework() = 0;  // i.e. the notify() method
  virtual ~Observer();
};

#endif

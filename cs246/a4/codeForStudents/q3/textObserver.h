#pragma once

#include "observer.h"
#include "studio.h"
#include <iostream>

class TextDisplay : public Observer {
private:
  const Studio &studio_;
  int top_, bottom_, left_, right_;
  std::ostream &out = std::cout;

public:
  TextDisplay(const Studio &studio, int top, int bottom, int left, int right);
  void notify() override;
};

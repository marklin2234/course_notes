#pragma once

#include "observer.h"
#include "studio.h"
#include "window.h"
#include <memory>

class GraphicDisplay : public Observer {
private:
  static constexpr int SQUARE_SIZE = 10;
  const Studio &studio_;
  int top_, bottom_, left_, right_;
  std::unique_ptr<Xwindow> win;

public:
  GraphicDisplay(const Studio &studio, int top, int bottom, int left,
                 int right);
  void notify() override;
};

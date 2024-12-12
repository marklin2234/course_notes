#pragma once

#include "Decorator.h"
#include <asciiart.h>

class MovingBoxDecorator : public Decorator {
private:
  int top_, bottom_, left_, right_;
  char symbol_;

public:
  MovingBoxDecorator(AsciiArt *art, int top, int bottom, int left, int right,
                     char symbol);
  char charAt(int row, int col, int tick) override;
};

#pragma once

#include "Decorator.h"
#include <asciiart.h>

class MaskBoxDecorator : public Decorator {
private:
  int top_, bottom_, left_, right_;
  char symbol_;

public:
  MaskBoxDecorator(AsciiArt *art, int top, int bottom, int left, int right,
                   char symbol);
  char charAt(int row, int col, int tick) override;
};

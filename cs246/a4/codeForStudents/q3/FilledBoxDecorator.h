#pragma once

#include "Decorator.h"
#include "asciiart.h"

class FilledBoxDecorator : public Decorator {
private:
  int top_, bottom_, left_, right_;
  char symbol_;

public:
  FilledBoxDecorator(AsciiArt *art, int top, int bottom, int left, int right,
                     char symbol);
  char charAt(int row, int col, int tick) override;
};

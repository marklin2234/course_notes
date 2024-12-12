#include "MovingBoxDecorator.h"
#include <Decorator.h>
#include <asciiart.h>

MovingBoxDecorator::MovingBoxDecorator(AsciiArt *art, int top, int bottom,
                                       int left, int right, char symbol)
    : Decorator{art}, top_{top}, bottom_{bottom}, left_{left}, right_{right},
      symbol_{symbol} {}

char MovingBoxDecorator::charAt(int row, int col, int tick) {
  if (row + tick >= top_ && row + tick <= bottom_ && col + tick >= left_ &&
      col + tick <= right_) {
    return symbol_;
  }

  return art_->charAt(row, col, tick);
}

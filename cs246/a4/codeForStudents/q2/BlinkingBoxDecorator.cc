#include "BlinkingBoxDecorator.h"
#include <Decorator.h>
#include <asciiart.h>

BlinkingBoxDecorator::BlinkingBoxDecorator(AsciiArt *art, int top, int bottom,
                                           int left, int right, char symbol)
    : Decorator{art}, top_{top}, bottom_{bottom}, left_{left}, right_{right},
      symbol_{symbol} {}

char BlinkingBoxDecorator::charAt(int row, int col, int tick) {
  if (tick % 2 == 0)
    return ' ';
  if (row >= top_ && row <= bottom_ && col >= left_ && col <= right_) {
    return symbol_;
  }

  return art_->charAt(row, col, tick);
}

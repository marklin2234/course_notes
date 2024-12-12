#include "FilledBoxDecorator.h"
#include "Decorator.h"
#include "asciiart.h"

FilledBoxDecorator::FilledBoxDecorator(AsciiArt *art, int top, int bottom,
                                       int left, int right, char symbol)
    : Decorator{art}, top_{top}, bottom_{bottom}, left_{left}, right_{right},
      symbol_{symbol} {}

char FilledBoxDecorator::charAt(int row, int col, int tick) {
  if (row >= top_ && row <= bottom_ && col >= left_ && col <= right_) {
    return symbol_;
  }

  return art_->charAt(row, col, tick);
}

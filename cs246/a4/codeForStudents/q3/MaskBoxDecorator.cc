#include "MaskBoxDecorator.h"
#include "Decorator.h"
#include "asciiart.h"

MaskBoxDecorator::MaskBoxDecorator(AsciiArt *art, int top, int bottom, int left,
                                   int right, char symbol)
    : Decorator{art}, top_{top}, bottom_{bottom}, left_{left}, right_{right},
      symbol_{symbol} {}

char MaskBoxDecorator::charAt(int row, int col, int tick) {
  char underneath = art_->charAt(row, col, tick);
  if (row >= top_ && row <= bottom_ && col >= left_ && col <= right_ && underneath != ' ') {
      return symbol_;
  }

  return underneath;
}

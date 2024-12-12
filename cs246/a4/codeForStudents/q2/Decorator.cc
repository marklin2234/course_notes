#include "Decorator.h"
#include <asciiart.h>

Decorator::Decorator(AsciiArt *art) : art_{art} {}
Decorator::~Decorator() { delete art_; }
char Decorator::charAt(int row, int col, int tick) {
  return art_->charAt(row, col, tick);
}

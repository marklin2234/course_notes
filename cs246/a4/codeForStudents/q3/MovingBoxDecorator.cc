#include "MovingBoxDecorator.h"
#include "Decorator.h"
#include "asciiart.h"

MovingBoxDecorator::MovingBoxDecorator(AsciiArt *art, int top, int bottom,
                                       int left, int right, char direction, char symbol)
    : Decorator{art}, top_{top}, bottom_{bottom}, left_{left}, right_{right},
      direction_{direction}, symbol_{symbol} {}

char MovingBoxDecorator::charAt(int row, int col, int tick) {
    int t = top_, b = bottom_, l = left_, r = right_;
    switch (direction_) {
        case 'u':
            t -= tick;
            b -= tick;
            break;
        case 'd':
            t += tick;
            b += tick;
            break;
        case 'l':
            l -= tick;
            r -= tick;
            break;
        case 'r':
            l += tick;
            r += tick;
            break;
    }
    if (row >= t && row <= b && col >= l && col <= r) {
        return symbol_;
    }

  return art_->charAt(row, col, tick);
}

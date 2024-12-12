#include "textObserver.h"

TextDisplay::TextDisplay(const Studio &studio, int top, int bottom, int left,
                         int right)
    : studio_{studio}, top_{top}, bottom_{bottom}, left_{left}, right_{right} {}

void TextDisplay::notify() {
  out << "+";
  for (int j = left_; j < right_; j++) {
    out << "-";
  }
  out << "+\n";
  for (int i = top_; i < bottom_; i++) {
    out << "|";
    for (int j = left_; j < right_; j++) {
      out << studio_.getState(i, j);
    }
    out << "|\n";
  }
  out << "+";
  for (int j = left_; j < right_; j++) {
    out << "-";
  }
  out << "+\n";
}

#include "graphicDisplay.h"
#include "window.h"

GraphicDisplay::GraphicDisplay(const Studio &studio, int top, int bottom,
                               int left, int right)
    : studio_{studio}, top_{top}, bottom_{bottom}, left_{left}, right_{right},
      win{std::make_unique<Xwindow>((bottom - top) * SQUARE_SIZE,
                                    (right - left) * SQUARE_SIZE)} {}

void GraphicDisplay::notify() {
  int rows = bottom_ - top_;
  int cols = right_ - left_;
  win->fillRectangle(0, 0, rows * SQUARE_SIZE, cols * SQUARE_SIZE,
                     Xwindow::White);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int y = i + top_, x = j + left_;
      char c = studio_.getState(y, x);

      if (c >= 97 && c <= 122) {
        win->fillRectangle(i * SQUARE_SIZE, j * SQUARE_SIZE, SQUARE_SIZE,
                           SQUARE_SIZE, Xwindow::Red);
      } else if (c >= 65 && c <= 90) {
        win->fillRectangle(i * SQUARE_SIZE, j * SQUARE_SIZE, SQUARE_SIZE,
                           SQUARE_SIZE, Xwindow::Green);
      } else if (c >= 48 && c <= 57) {
        win->fillRectangle(i * SQUARE_SIZE, j * SQUARE_SIZE, SQUARE_SIZE,
                           SQUARE_SIZE, Xwindow::Blue);
      } else if (c >= 33 && c <= 126) {
        win->fillRectangle(i * SQUARE_SIZE, j * SQUARE_SIZE, SQUARE_SIZE,
                           SQUARE_SIZE, Xwindow::Black);
      }
    }
  }
}

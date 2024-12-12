#include "BlinkingBoxDecorator.h"
#include "FilledBoxDecorator.h"
#include "MaskBoxDecorator.h"
#include "MovingBoxDecorator.h"
#include "asciiart.h"
#include "blank.h"
#include "graphicDisplay.h"
#include "studio.h"
#include "textObserver.h"
#include <iostream>
#include <vector>

int main() {
  AsciiArt *canvas = new Blank;

  Studio s{canvas};

  std::string command;

  while (std::cin >> command) {
    if (command == "render") {
      s.render();
    } else if (command == "animate") {
      int n;
      std::cin >> n;
      s.animate(n);
    } else if (command == "reset") {
      s.reset();
    } else if (command == "filledbox") {
      int top, bottom, left, right;
      char what;
      std::cin >> top >> bottom >> left >> right >> what;
      s.picture() =
          new FilledBoxDecorator(s.picture(), top, bottom, left, right, what);
    } else if (command == "blinkingbox") {
      int top, bottom, left, right;
      char what;
      std::cin >> top >> bottom >> left >> right >> what;
      s.picture() =
          new BlinkingBoxDecorator(s.picture(), top, bottom, left, right, what);
    } else if (command == "movingbox") {
      int top, bottom, left, right;
      char what, dir;
      std::cin >> top >> bottom >> left >> right >> what >> dir;
      s.picture() = new MovingBoxDecorator(s.picture(), top, bottom, left,
                                           right, dir, what);
    } else if (command == "maskbox") {
      int top, bottom, left, right;
      char what;
      std::cin >> top >> bottom >> left >> right >> what;
      s.picture() =
          new MaskBoxDecorator(s.picture(), top, bottom, left, right, what);
    } else if (command == "addtext") {
      int top, bottom, left, right;
      std::cin >> top >> bottom >> left >> right;
      s.attach(new TextDisplay(s, top, bottom, left, right));
    } else if (command == "addgraphics") {
      int top, bottom, left, right;
      std::cin >> top >> bottom >> left >> right;
      s.attach(new GraphicDisplay(s, top, bottom, left, right));
    } // if
  } // while
} // main
